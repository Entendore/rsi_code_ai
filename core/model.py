import tensorflow as tf
import numpy as np
import re
from collections import Counter
from config import Config

class BPETokenizer:
    def __init__(self, vocab_size=500):
        self.vocab_size = vocab_size
        self.vocab = []
        self.token_to_id = {}
        self.id_to_token = {}

    def train(self, texts):
        words = []
        for text in texts:
            words.extend(re.findall(r'\w+|[^\w\s]', text))
        chars = list(set("".join(words)))
        self.vocab = chars[:]
        word_counts = Counter(words)
        while len(self.vocab) < self.vocab_size:
            pairs = Counter()
            for word, count in word_counts.items():
                for i in range(len(word)-1): pairs[(word[i], word[i+1])] += count
            if not pairs: break
            best_pair = pairs.most_common(1)[0][0]
            self.vocab.append("".join(best_pair))
            new_word_counts = Counter()
            for word, count in word_counts.items():
                new_word_counts[word.replace("".join(best_pair), "".join(best_pair))] = count
            word_counts = new_word_counts
        self.token_to_id = {tok: i+1 for i, tok in enumerate(self.vocab)}
        self.id_to_token = {i+1: tok for i, tok in enumerate(self.vocab)}
        self.id_to_token[0] = "<PAD>"

    def encode(self, text, max_len):
        words = re.findall(r'\w+|[^\w\s]', text)
        ids = []
        for w in words:
            if w in self.token_to_id: ids.append(self.token_to_id[w])
            else:
                for c in w: ids.append(self.token_to_id.get(c, 0))
        if len(ids) < max_len: ids += [0] * (max_len - len(ids))
        return ids[:max_len]

    def decode(self, ids):
        return " ".join([self.id_to_token.get(int(i), "") for i in ids if i > 0])

class CustomCodeTransformer(tf.keras.Model):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tokenizer = BPETokenizer(config.VOCAB_SIZE)
        self.token_emb = tf.keras.layers.Embedding(config.VOCAB_SIZE, config.EMBED_DIM)
        self.pos_emb = tf.keras.layers.Embedding(config.MAX_SEQ_LEN, config.EMBED_DIM)
        self.att1 = tf.keras.layers.MultiHeadAttention(num_heads=config.NUM_HEADS, key_dim=config.EMBED_DIM)
        self.ffn1 = tf.keras.Sequential([tf.keras.layers.Dense(config.FF_DIM, activation='relu'), tf.keras.layers.Dense(config.EMBED_DIM)])
        self.layernorm1 = tf.keras.layers.LayerNormalization()
        self.layernorm2 = tf.keras.layers.LayerNormalization()
        self.dense_out = tf.keras.layers.Dense(config.VOCAB_SIZE)
        self.optimizer_sup = tf.keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LEARNING_RATE)
        self.optimizer_rl = tf.keras.optimizers.Adam(learning_rate=config.RL_LEARNING_RATE)
        self.fisher_matrix = None
        self.optimal_weights = None

    def call(self, inputs, training=False):
        seq_len = tf.shape(inputs)[1]
        positions = tf.range(start=0, limit=seq_len, delta=1)
        x = self.token_emb(inputs) + self.pos_emb(positions)
        attn_output = self.att1(x, x, x, use_causal_mask=True)
        out1 = self.layernorm1(x + attn_output)
        ffn_output = self.ffn1(out1)
        out2 = self.layernorm2(out1 + ffn_output)
        return self.dense_out(out2)

    def generate_with_confidence(self, prompt, beam_width=3, max_tokens=50):
        input_ids = self.tokenizer.encode(prompt, self.config.MAX_SEQ_LEN)
        initial_tensor = tf.constant([input_ids], dtype=tf.int32)
        sequences = [(initial_tensor, 0.0)]
        for _ in range(max_tokens):
            all_candidates = []
            for seq_tensor, score in sequences:
                context = seq_tensor[:, -self.config.MAX_SEQ_LEN:]
                logits = self(context, training=False)
                next_logits = logits[0, -1, :]
                top_k_logits, top_k_indices = tf.math.top_k(next_logits, k=beam_width)
                top_k_log_probs = tf.nn.log_softmax(top_k_logits)
                for i in range(beam_width):
                    next_token = tf.expand_dims(tf.expand_dims(top_k_indices[i], 0), 0)
                    new_seq = tf.concat([seq_tensor, next_token], axis=1)
                    new_score = score + top_k_log_probs[i].numpy()
                    all_candidates.append((new_seq, new_score))
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]
            if all([tf.equal(seq[0, -1], 0) for seq, _ in sequences]): break
        best_seq = sequences[0][0][0].numpy()
        best_score = sequences[0][1]
        generated_text = self.tokenizer.decode(best_seq)
        confidence = float(tf.sigmoid(best_score / 10.0).numpy())
        return generated_text[len(prompt):].strip(), confidence

    def supervised_fine_tune(self, prompt, successful_code):
        text = prompt + " " + successful_code
        input_ids = self.tokenizer.encode(text, self.config.MAX_SEQ_LEN)
        input_tensor = tf.constant([input_ids], dtype=tf.int32)
        with tf.GradientTape() as tape:
            logits = self(input_tensor, training=True)
            shift_logits = logits[:, :-1, :]
            shift_labels = input_tensor[:, 1:]
            loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
            loss = loss_fn(shift_labels, shift_logits)
            if self.fisher_matrix is not None:
                ewc_penalty = 0.0
                for i in range(len(self.trainable_variables)):
                    diff = self.trainable_variables[i] - self.optimal_weights[i]
                    ewc_penalty += tf.reduce_sum(self.fisher_matrix[i] * tf.square(diff))
                loss += self.config.EWC_LAMBDA * ewc_penalty
        gradients = tape.gradient(loss, self.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.config.GRADIENT_CLIP_NORM)
        self.optimizer_sup.apply_gradients(zip(gradients, self.trainable_variables))
        return loss.numpy()

class DPOTrainer:
    def __init__(self, model, beta=0.1, learning_rate=5e-5):
        self.model = model
        self.beta = beta
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def _get_logps(self, prompt, code):
        text = prompt + " " + code
        input_ids = self.model.tokenizer.encode(text, self.model.config.MAX_SEQ_LEN)
        input_tensor = tf.constant([input_ids], dtype=tf.int32)
        logits = self.model(input_tensor, training=True)
        shift_logits = logits[:, :-1, :]
        shift_labels = input_tensor[:, 1:]
        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
        neg_log_probs = loss_fn(shift_labels, shift_logits)
        return tf.reduce_mean(neg_log_probs)

    def train_step(self, prompt, chosen_code, rejected_code):
        with tf.GradientTape() as tape:
            pi_logratios_chosen = -self._get_logps(prompt, chosen_code)
            pi_logratios_rejected = -self._get_logps(prompt, rejected_code)
            logits = self.beta * (pi_logratios_chosen - pi_logratios_rejected)
            loss = -tf.math.log(tf.sigmoid(logits) + 1e-8)
            loss = tf.reduce_mean(loss)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        gradients, _ = tf.clip_by_global_norm(gradients, self.model.config.GRADIENT_CLIP_NORM)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss.numpy()
