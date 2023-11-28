import gzip
import os
import html
import ftfy
import re
import string
import tensorflow as tf
import tensorflow_text as tf_text
from keras.src.utils import data_utils
from tensorflow_hub import KerasLayer

DEFAULT_CONTEXT_LENGTH = 77  # Default context length for OpenAI CLIP


def basic_clean(text):
    text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    text = text.strip()

    return text


class SimpleTokenizer:
    def __init__(self, context_length=DEFAULT_CONTEXT_LENGTH):
        self.context_length = context_length

        byte_list, unicode_list = self._bytes_to_unicode()

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

    def _bytes_to_unicode(self):
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
            inputs = tf.convert_to_tensor(inputs, 'string')
            if 1 != inputs.shape.rank:
                raise ValueError(f'Expecting a vector of strings. Got shape {inputs.shape}')

            # Lowercase input sentences and separate words with space
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
                lambda: cache_lookup)

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
        chars = chars[:, :self.max_len * 2]  # double max length due end-of-word marker changes merge priority

        # Map bytes to unicode
        chars = self.byte2unicode.lookup(chars)

        # Add end-of-word marker
        chars = tf.concat([
            chars[:, :-1], tf.strings.join([chars[:, -1:], '</w>'])], axis=-1)

        # Merge characters
        tokens, _ = tf.while_loop(
            lambda _, mask: tf.math.reduce_any(mask),
            self._bpe_step,
            loop_vars=(chars, tf.ones_like(words, 'bool')),
            shape_invariants=(tf.TensorShape([None, None]), tf.TensorShape([None])))

        return tokens

    @tf.function
    def _bpe_step(self, words, merge_mask):
        merge_mask &= words.row_lengths() > 1

        return tf.cond(
            tf.reduce_any(merge_mask),
            lambda: self._bpe_merge(words, merge_mask),
            lambda: (words, merge_mask))

    def _bpe_merge(self, words, merge_mask):
        # Sub-word pairs for words that possible can be merged
        curr_words = tf.ragged.boolean_mask(words, merge_mask)
        pairs_left, pairs_right = curr_words[:, :-1], curr_words[:, 1:]
        curr_pairs = tf.strings.join([pairs_left, pairs_right], separator=' ')

        # Byte pair ranking in merge rules
        curr_ranks = self.merge2rank.lookup(curr_pairs)
        min_ranks = tf.reduce_min(curr_ranks, axis=-1)
        curr_found = min_ranks < self.merge_size + 1

        # Tokens that cannot be further merged are marked as finished
        found_indices = tf.boolean_mask(tf.range(tf.size(merge_mask)), merge_mask)
        merge_mask = tf.tensor_scatter_nd_update(merge_mask, found_indices[:, None], curr_found)

        return tf.cond(
            tf.reduce_any(merge_mask),
            lambda: self._bpe_replace(words, merge_mask, curr_ranks, min_ranks, curr_found),
            lambda: (words, merge_mask))

    def _bpe_replace(self, words, merge_mask, pair_ranks, min_ranks, found_mask):
        # Mask for right sub-word in pairs that definitely should be merged
        merge_left = tf.equal(pair_ranks, min_ranks[:, None])
        merge_left = tf.ragged.boolean_mask(merge_left, found_mask)
        merge_shift = tf.zeros([merge_left.nrows(), 1], 'bool')
        merge_right = tf.concat([merge_shift, merge_left], axis=1)

        # All sub-words indices
        words_size = tf.size(words.flat_values)
        words_range = tf.range(words_size)
        word_idx = words.with_flat_values(words_range)

        # Separate `skip` and `merge` sub-word indices
        curr_idx = tf.ragged.boolean_mask(word_idx, merge_mask)
        skip_idx = tf.ragged.boolean_mask(word_idx, ~merge_mask)

        # Segment shift from sub-word merge mask
        seg_shift = words.with_flat_values(tf.dynamic_stitch(
            [curr_idx.flat_values, skip_idx.flat_values],
            [tf.cast(merge_right.flat_values, words_size.dtype), tf.zeros_like(skip_idx.flat_values)]))

        # Merge sub-word pairs
        segment_ids = words_range - tf.cumsum(seg_shift.flat_values)
        num_segments = tf.reduce_max(segment_ids) + 1
        merged_words = tf.strings.unsorted_segment_join(words.flat_values, segment_ids, num_segments, '')

        # Merged sub-words
        words_lengths = words.row_lengths()
        merged_lengths = words_lengths - tf.cast(tf.reduce_sum(seg_shift, axis=-1), words_lengths.dtype)
        words = tf.RaggedTensor.from_row_lengths(merged_words, merged_lengths)

        return words, merge_mask


class SentencePieceTokenizer:
    def __init__(self, tokenizer_name, context_length=DEFAULT_CONTEXT_LENGTH, clean=None):
        model_url = f'https://github.com/shkarupa-alex/tfclip/releases/download/{tokenizer_name}.model'
        model_hash = model_url.split('__')[-1].replace('.model', '')
        model_path = data_utils.get_file(origin=model_url, file_hash=model_hash, cache_subdir='tfclip')

        with tf.io.gfile.GFile(model_path, 'rb') as f:
            model = f.read()

        self.tokenizer = tf_text.SentencepieceTokenizer(model)
        self.context_length = context_length

        self.canonicalize = None
        if 'canonicalize' == clean:
            punctuation = re.escape(string.punctuation.replace(',', ''))
            self.canonicalize = f'[{punctuation}]+'

    def __call__(self, inputs):
        with tf.device('cpu'):
            inputs = tf.convert_to_tensor(inputs, 'string')
            if 1 != inputs.shape.rank:
                raise ValueError(f'Expecting a vector of strings. Got shape {inputs.shape}')

            # https://github.com/google-research/big_vision/issues/79
            inputs = tf.strings.lower(inputs)  # , encoding='utf-8'

            if self.canonicalize:
                inputs = tf.strings.regex_replace(inputs, self.canonicalize, '')

            tokens = self.tokenizer.tokenize(inputs)

            if self.context_length:
                output_shape = tokens.shape.as_list()
                output_shape[-1] = self.context_length - 1
                tokens = tokens[:, :self.context_length - 1]
                tokens = tokens.to_tensor(default_value=1, shape=output_shape)
                tokens.set_shape(output_shape)
                tokens = tf.pad(tokens, [[0, 0], [0, 1]], constant_values=1)

            return tokens


class TensorflowHubTokenizer:
    def __init__(self, tokenizer_name, context_length=DEFAULT_CONTEXT_LENGTH, strip_sep_token=False):
        self.tokenizer = KerasLayer(f'https://tfhub.dev/tensorflow/{tokenizer_name}')
        self.context_length = context_length
        self.strip_sep_token = strip_sep_token

    def __call__(self, inputs):
        with tf.device('cpu'):
            inputs = tf.convert_to_tensor(inputs, 'string')
            if 1 != inputs.shape.rank:
                raise ValueError(f'Expecting a vector of strings. Got shape {inputs.shape}')

            tokens = self.tokenizer(inputs)['input_word_ids']

            if self.strip_sep_token:
                tokens = tf.where(tf.equal(tokens, int(self.strip_sep_token)), 0, tokens)

            if self.context_length:
                output_shape = tokens.shape.as_list()
                output_shape[-1] = self.context_length - 1
                tokens = tokens[:, :self.context_length - 1]
                tokens.set_shape(output_shape)
                tokens = tf.pad(tokens, [[0, 0], [0, 1]])

            return tokens
