import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Layer, Softmax, Multiply
from tensorflow.keras.models import Model


class Gate(Layer):
    def __init__(self, num_experts, **kwargs):
        super(Gate, self).__init__(**kwargs)
        # self.num_experts = num_experts
        self.hidden = 20
        self.gate_dropout = tf.keras.layers.Dropout(0.3)
        self.dense1 = Dense(self.hidden, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense2 = Dense(self.hidden, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))
        self.dense3 = Dense(num_experts, activation='softmax', kernel_regularizer=tf.keras.regularizers.l2(0.01))

    def call(self, inputs):
        out = self.dense1(inputs)
        out = self.gate_dropout(out)
        out = self.dense2(out)
        out = self.gate_dropout(out)
        out = self.dense3(out)
        return out


class MixtureOfExperts(Model):
    def __init__(self, experts, num_experts, output_units, **kwargs):
        super(MixtureOfExperts, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.experts = [Expert for Expert in experts]
        self.gate = Gate(num_experts)
        self.output_layer = Dense(output_units)

    def call(self, inputs):
        expert_outputs = [self.experts[0](inputs), self.experts[1](inputs), self.experts[2](inputs),
                          self.experts[3](inputs)[:, 0:1]]

        # print(self.experts[3])
        # print(self.experts[3](inputs))
        # print(self.experts[3](inputs)[:,0:1])
        expert_outputs = tf.stack(expert_outputs, axis=1)

        gate_outputs = self.gate(inputs)
        gate_outputs = tf.expand_dims(gate_outputs, axis=2)

        weighted_expert_outputs = Multiply()([expert_outputs, gate_outputs])
        mixed_output = tf.reduce_sum(weighted_expert_outputs, axis=1)

        return mixed_output


class MixtureOfExperts_multi_experts(Model):
    def __init__(self, experts, num_experts, output_units, **kwargs):
        super(MixtureOfExperts_multi_experts, self).__init__(**kwargs)
        self.num_experts = num_experts
        self.experts = experts
        self.gate = Gate(num_experts)
        self.output_layer = Dense(output_units)
        self.topK = True

    def call(self, inputs):
        env = inputs[0]
        experts_predictions = inputs[1]
        size_experts=experts_predictions.shape[1]
        expert_outputs = [tf.expand_dims(self.experts[0](env),axis=1), self.experts[1](env), self.experts[2](env), self.experts[3](env),
                          self.experts[4](env)[:, 0:1], tf.expand_dims(experts_predictions[:, 0], axis=1),
                          tf.expand_dims(experts_predictions[:, 1], axis=1),
                          tf.expand_dims(experts_predictions[:, 2], axis=1)]
        # print(self.experts[0](env))
        # print(self.experts[1](env))
        # expert_outputs = [tf.expand_dims(self.experts[0](env),axis=1), self.experts[1](env), self.experts[2](env),
        #                   self.experts[3](env), self.experts[4](env)[:, 0:1], experts_predictions]

        # print(experts_predictions)
        # print(experts_predictions[:,0])
        # print(self.experts[3](env)[:, 0:1])
        expert_outputs = tf.stack(expert_outputs, axis=1)

        gate_outputs = self.gate(env)
        # print(gate_outputs)
        # print(expert_outputs)
        #gate_outputs = tf.expand_dims(gate_outputs, axis=2)

        if self.topK:
            K = 3
            _, top_k_indices = tf.nn.top_k(gate_outputs, k=K)
            top_k_outputs = tf.gather(expert_outputs, top_k_indices,axis=1,batch_dims=True)
            top_k_weights = tf.gather(gate_outputs, top_k_indices,axis=1,batch_dims=True)
            # print(top_k_indices)
            top_k_weights = tf.nn.softmax(top_k_weights, axis=1)
            mixed_output = tf.reduce_sum(tf.squeeze(top_k_outputs) * top_k_weights, axis=1)
            # print(mixed_output)
        else:
            gate_outputs = tf.expand_dims(gate_outputs, axis=2)
            # print(expert_outputs)
            # print(gate_outputs)
            weighted_expert_outputs = Multiply()([expert_outputs, gate_outputs])
            mixed_output = tf.reduce_sum(weighted_expert_outputs, axis=1)
            # print(mixed_output)

        return mixed_output

    # def train_step(self, x, y, additional_data):
    #     # 与额外数据合并的主输入
    #     inputs_with_additional_data = tf.concat([x, additional_data], axis=-1)
    #
    #     with tf.GradientTape() as tape:
    #         # 计算模型输出
    #         logits = self(inputs_with_additional_data, training=True)
    #         # 计算损失
    #         loss_value = self.loss(y, logits)
    #
    #     # 计算梯度
    #     gradients = tape.gradient(loss_value, self.trainable_variables)
    #     # 应用梯度更新
    #     self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
    #
    #     # 返回损失值和额外的指标（如果有的话）
    #     return loss_value
