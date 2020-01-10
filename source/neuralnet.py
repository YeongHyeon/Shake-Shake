import tensorflow as tf

class ShakeShakeNet(object):

    def __init__(self, height, width, channel, num_class, leaning_rate=1e-3):

        print("\nInitializing Neural Network...")
        self.height, self.width, self.channel = height, width, channel
        self.num_class, self.k_size = num_class, 3
        self.leaning_rate = leaning_rate

        self.x = tf.compat.v1.placeholder(tf.float32, [None, self.height, self.width, self.channel])
        self.y = tf.compat.v1.placeholder(tf.float32, [None, self.num_class])
        self.batch_size = tf.compat.v1.placeholder(tf.int32, shape=[])

        self.weights, self.biasis = [], []
        self.w_names, self.b_names = [], []

        self.y_hat = self.build_model(input=self.x)

        self.smce = tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.y_hat)
        self.loss = tf.compat.v1.reduce_mean(self.smce)

        #default: beta1=0.9, beta2=0.999
        self.optimizer = tf.compat.v1.train.AdamOptimizer( \
            self.leaning_rate, beta1=0.9, beta2=0.999).minimize(self.loss)

        self.score = tf.nn.softmax(self.y_hat)
        self.pred = tf.argmax(self.score, 1)
        self.correct_pred = tf.equal(self.pred, tf.argmax(self.y, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))

        tf.compat.v1.summary.scalar('softmax_cross_entropy', self.loss)
        self.summaries = tf.compat.v1.summary.merge_all()

    def build_model(self, input):

        conv1_1 = self.conv2d(input=input, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, 1, 16], activation="relu", name="conv1_1")
        conv1_2 = self.conv2d(input=conv1_1, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, 16, 16], activation="relu", name="conv1_2")
        max_pool1 = self.maxpool(input=conv1_2, ksize=2, strides=2, padding='SAME', name="max_pool1")

        conv2_1 = self.conv2d(input=max_pool1, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, 16, 32], activation="relu", name="conv2_1")
        conv2_2 = self.conv2d(input=conv2_1, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, 32, 32], activation="relu", name="conv2_2")
        max_pool2 = self.maxpool(input=conv2_2, ksize=2, strides=2, padding='SAME', name="max_pool2")

        conv3_1 = self.conv2d(input=max_pool2, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, 32, 64], activation="relu", name="conv3_1")
        conv3_2 = self.conv2d(input=conv3_1, stride=1, padding='SAME', \
            filter_size=[self.k_size, self.k_size, 64, 64], activation="relu", name="conv3_2")

        [n, h, w, c] = conv3_2.shape
        fullcon_in = tf.compat.v1.reshape(conv3_2, shape=[self.batch_size, h*w*c], name="fullcon_in")
        fullcon1 = self.fully_connected(input=fullcon_in, num_inputs=int(h*w*c), \
            num_outputs=512, activation="relu", name="fullcon1")
        fullcon2 = self.fully_connected(input=fullcon1, num_inputs=512, \
            num_outputs=self.num_class, activation=None, name="fullcon2")

        return fullcon2

    def initializer(self):
        return tf.compat.v1.initializers.variance_scaling(distribution="untruncated_normal", dtype=tf.dtypes.float32)

    def maxpool(self, input, ksize, strides, padding, name=""):

        out_maxp = tf.compat.v1.nn.max_pool(value=input, \
            ksize=ksize, strides=strides, padding=padding, name=name)
        print("Max-Pool", input.shape, "->", out_maxp.shape)

        return out_maxp

    def activation_fn(self, input, activation="relu", name=""):

        if("sigmoid" == activation):
            out = tf.compat.v1.nn.sigmoid(input, name='%s_sigmoid' %(name))
        elif("tanh" == activation):
            out = tf.compat.v1.nn.tanh(input, name='%s_tanh' %(name))
        elif("relu" == activation):
            out = tf.compat.v1.nn.relu(input, name='%s_relu' %(name))
        elif("lrelu" == activation):
            out = tf.compat.v1.nn.leaky_relu(input, name='%s_lrelu' %(name))
        elif("elu" == activation):
            out = tf.compat.v1.nn.elu(input, name='%s_elu' %(name))
        else: out = input

        return out

    def variable_maker(self, var_bank, name_bank, shape, name=""):

        try:
            var_idx = name_bank.index(name)
        except:
            variable = tf.compat.v1.get_variable(name=name, \
                shape=shape, initializer=self.initializer())

            var_bank.append(variable)
            name_bank.append(name)
        else:
            variable = var_bank[var_idx]

        return var_bank, name_bank, variable

    def conv2d(self, input, stride, padding, \
        filter_size=[3, 3, 16, 32], dilations=[1, 1, 1, 1], activation="relu", name=""):

        # strides=[N, H, W, C], [1, stride, stride, 1]
        # filter_size=[ksize, ksize, num_inputs, num_outputs]
        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=filter_size, name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[filter_size[-1]], name='%s_b' %(name))

        out_conv = tf.compat.v1.nn.conv2d(
            input=input,
            filter=weight,
            strides=[1, stride, stride, 1],
            padding=padding,
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=dilations,
            name='%s_conv' %(name),
        )
        out_bias = tf.math.add(out_conv, bias, name='%s_add' %(name))

        print("Conv", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)

    def fully_connected(self, input, num_inputs, num_outputs, activation="relu", name=""):

        self.weights, self.w_names, weight = self.variable_maker(var_bank=self.weights, name_bank=self.w_names, \
            shape=[num_inputs, num_outputs], name='%s_w' %(name))
        self.biasis, self.b_names, bias = self.variable_maker(var_bank=self.biasis, name_bank=self.b_names, \
            shape=[num_outputs], name='%s_b' %(name))

        out_mul = tf.compat.v1.matmul(input, weight, name='%s_mul' %(name))
        out_bias = tf.math.add(out_mul, bias, name='%s_add' %(name))

        print("Full-Con", input.shape, "->", out_bias.shape)
        return self.activation_fn(input=out_bias, activation=activation, name=name)
