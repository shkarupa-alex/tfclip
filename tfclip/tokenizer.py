import gzip
import os
import html
import ftfy
import tensorflow as tf

DEFAULT_CONTEXT_LENGTH = 77  # Default context length for OpenAI CLIP


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()

    return text


class SimpleTokenizer:
    def __init__(self, context_length=DEFAULT_CONTEXT_LENGTH):
        self.context_length = context_length

        byte_list, unicode_list = SimpleTokenizer.bytes_to_unicode()

        vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'bpe_simple_vocab_16e6.txt.gz')
        with gzip.open(vocab_path, 'rt') as vocab_file:
            merges = vocab_file.read().split('\n')
        merges = merges[1:49152 - 256 - 2 + 1]
        self.merge_size = len(merges)
        self.max_len = max([len(m.replace(' ', '').replace('</w>', '').encode('utf-8')) for m in merges])

        vocabulary = unicode_list + [f'{u}</w>' for u in unicode_list]
        vocabulary.extend([m.replace(' ', '') for m in merges])
        self.vocab_size = len(vocabulary)

        # Byte -> unicode mapping
        self.byte2unicode = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(byte_list, unicode_list), '')

        # Token -> id mapping
        self.token2id = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(vocabulary, range(len(vocabulary))), -1)

        # Pair -> frequency mapping
        self.merge2rank = tf.lookup.StaticHashTable(
            tf.lookup.KeyValueTensorInitializer(merges, range(len(merges))), self.merge_size + 1)

        # Word -> id cache
        # `tf.lookup.experimental.MutableHashTable` does not support string to string mapping.
        # So we first convert to string to an integer key, and use the integer key to find the value.
        self.cache_factors = tf.pow(tf.constant(256, dtype='int64'), tf.range(0, 8, dtype='int64'))[:, None]
        self.cache_id2value = tf.lookup.experimental.MutableHashTable('int64', 'string', '')

    @staticmethod
    def bytes_to_unicode():
        bs = list(range(ord("!"), ord("~") + 1)) + \
             list(range(ord("¡"), ord("¬") + 1)) + \
             list(range(ord("®"), ord("ÿ") + 1))
        cs = bs[:]
        n = 0
        for b in range(2 ** 8):
            if b not in bs:
                bs.append(b)
                cs.append(2 ** 8 + n)
                n += 1
        cs = [chr(n) for n in cs]
        bs = [n.to_bytes(1, "little") for n in bs]

        return bs, cs

    def __call__(self, inputs):
        with tf.device('cpu'):
            # Lowercase input sentences and separate words with space
            inputs = tf.convert_to_tensor(inputs, 'string')
            inputs = tf.strings.lower(inputs, encoding='utf-8')
            inputs = tf.strings.regex_replace(
                inputs, '[\v\u000b\u0085\u00a0\u0345\u1680\u2000-\u200a\u2028\u2029\u202f\u205f\u3000]', ' ')
            inputs = tf.strings.regex_replace(
                inputs, r"'s|'t|'re|'ve|'m|'ll|'d|\p{L}+|\p{N}|[^\s\p{L}\p{N}]+", r" \0 ")

            # Split sentences to words
            raw_tokens = tf.strings.split(inputs)

            # Slice long sentences to speedup
            raw_tokens = raw_tokens[..., :self.context_length - 2]

            row_splits = raw_tokens.row_splits
            flat_tokens = raw_tokens.flat_values

            # Get word indices from cache
            cache_lookup = self._cache_lookup(flat_tokens)
            cache_miss = tf.equal(cache_lookup, '') & tf.not_equal(flat_tokens, '')

            # Tokenize unseen words
            tokenized_words = tf.cond(
                tf.math.reduce_any(cache_miss),
                lambda: self._process_unseen(flat_tokens, cache_miss),
                lambda: tf.identity(cache_lookup),
            )

            # Lookup token indices
            tokens = tf.strings.split(tokenized_words)
            tokens = self.token2id.lookup(tokens)

            # Unflatten to match input
            tokens = tf.RaggedTensor.from_row_splits(tokens.flat_values, tf.gather(tokens.row_splits, row_splits))

            # Slice to required sequence length
            if self.context_length:
                tokens = tokens[:, :self.context_length - 2]

            # Add begin-of-sentence and begin-of-sentence tokens
            bos = tf.fill([tokens.nrows(), 1], self.vocab_size)
            eos = tf.fill([tokens.nrows(), 1], self.vocab_size + 1)
            tokens = tf.concat([bos, tokens, eos], axis=1)

            # Make indices dense
            if self.context_length:
                output_shape = tokens.shape.as_list()
                output_shape[-1] = self.context_length
                tokens = tokens.to_tensor(shape=output_shape)

            return tokens

    def _cache_key(self, keys):
        # `tf.fingerprint` converts token to an array of uint8 of length 8, we need to convert it to int64.
        return tf.squeeze(tf.matmul(tf.cast(tf.fingerprint(keys), dtype='int64'), self.cache_factors), -1)

    def _cache_lookup(self, keys):
        ids = self._cache_key(keys)
        result = self.cache_id2value.lookup(ids)
        result.set_shape([None])  # Ensure output shape for graph mode.

        return result

    def _cache_insert(self, keys, values):
        self.cache_id2value.insert(self._cache_key(keys), values)

    def _process_unseen(self, tokens, miss_mask):
        # Process unseen tokens and add to cache
        unseen_tokens = tf.boolean_mask(tokens, miss_mask)
        seen_words = self._bpe_encode(unseen_tokens)
        seen_words = tf.strings.reduce_join(seen_words, axis=1, separator=' ')
        self._cache_insert(unseen_tokens, seen_words)

        return self._cache_lookup(tokens)

    def _bpe_encode(self, words):
        # Split to bytes
        chars = tf.strings.bytes_split(words)

        # Slice long words to speedup
        chars = chars[:, :self.max_len * 2]  # Double max length due end-of-word marker changes merge priority

        # Map bytes to unicode
        chars = self.byte2unicode.lookup(chars)

        # Add end-of-word marker
        chars = tf.concat([
            chars[:, :-1], tf.strings.join([chars[:, -1:], '</w>'])], axis=-1)

        # Merge characters
        tokens, _ = tf.while_loop(
            lambda _, mask: tf.math.reduce_any(mask),
            self._bpe_step,
            loop_vars=[chars, tf.fill(tf.shape(words), True)],
            shape_invariants=[tf.TensorShape([None, None]), tf.TensorShape([None])])

        return tokens

    @tf.function
    def _bpe_step(self, words, merge_mask):
        # Prepare sub-word neighboring pairs
        pairs_left, pairs_right = words[:, :-1], words[:, 1:]

        non_empty = tf.not_equal(pairs_right.nested_row_lengths()[0], 0)
        merge_mask = merge_mask & non_empty

        return tf.cond(
            tf.reduce_any(merge_mask),
            lambda: self._bpe_merge(words, merge_mask, pairs_left, pairs_right),
            lambda: [words, merge_mask])

    def _bpe_merge(self, words, merge_mask, pairs_left, pairs_right):
        pairs_left = tf.ragged.boolean_mask(pairs_left, merge_mask)
        pairs_right = tf.ragged.boolean_mask(pairs_right, merge_mask)

        # Byte pair ranking in merge rules
        pairs = tf.strings.join([pairs_left, pairs_right], separator=' ')
        ranks = self.merge2rank.lookup(pairs)
        min_ranks = tf.reduce_min(ranks, axis=1)
        found_mask = tf.not_equal(min_ranks, self.merge_size + 1)

        # Tokens that cannot be further merged are marked as finished
        words_range = tf.range(tf.shape(merge_mask)[0])
        found_indices = tf.boolean_mask(words_range, merge_mask)
        merge_mask = tf.tensor_scatter_nd_update(merge_mask, tf.expand_dims(found_indices, axis=1), found_mask)

        return tf.cond(
            tf.reduce_any(merge_mask),
            lambda: self._bpe_replace(words, merge_mask, ranks, found_mask, words_range, min_ranks),
            lambda: [words, merge_mask])

    def _bpe_replace(self, words, merge_mask, pair_rank, found_mask, words_range):
        pair_rank = tf.ragged.boolean_mask(pair_rank, found_mask)
        top_indices = tf.math.argmin(pair_rank.to_tensor(self.merge_size + 1), axis=1)

        # Words and pairs to merge
        unfinished_words = tf.ragged.boolean_mask(words, merge_mask)
        pair_left = tf.gather(unfinished_words, top_indices, batch_dims=1)
        pair_right = tf.gather(unfinished_words, top_indices + 1, batch_dims=1)

        # Concatenate pairs and prepare replacement
        merged_pairs = tf.strings.join([pair_left, pair_right])
        empty_words = tf.fill(tf.shape(merged_pairs), '')

        # Sampling indices
        unfinished_indices = tf.cast(tf.boolean_mask(words_range, merge_mask), dtype='int64')
        merged_indices = tf.stack([unfinished_indices, top_indices], axis=1)
        empty_indices = tf.stack([unfinished_indices, top_indices + 1], axis=1)

        # Merged pairs
        dense_words = words.to_tensor(default_value='')
        dense_words = tf.tensor_scatter_nd_update(dense_words, merged_indices, merged_pairs)

        # Replace free positions with empty string
        words = tf.tensor_scatter_nd_update(dense_words, empty_indices, empty_words)

        # Drop empty strings
        words = tf.strings.split(tf.strings.reduce_join(words, separator=' ', axis=-1))

        return [words, merge_mask]
