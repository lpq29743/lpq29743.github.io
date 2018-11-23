开启 TensorBoard

```bash
# --logdir 参数表示日志目录
tensorboard --logdir=logs
# --port 可以设置端口，默认端口为 6006
tensorboard --logdir=logs --port=6007
```

TensorFlow 下的视图：

1. Scalars：可视化标量值，如训练集、验证集或测试集的准确率和损失
2. Graph：可视化计算图，借助`tf.name_scope`定义图会辅助理解，增加计算图的可读性。Graph 有两种配色方案：默认的结构视图显示了结构：当两个高级别节点具有相同的结构时，它们将显示相同的彩虹色。具有独特结构的节点显示为灰色。第二个视图显示运行不同指令的设备。名称范围根据其内部指令的设备比例来按比例着色。
3. Distributions：可视化数据随着时间的变化，如神经网络中的权重
4. Histograms：从 3 维的角度展示分布
5. Projector：可用来可视化词向量
6. Image：可视化图片数据
7. Audio：可视化音频数据
8. Text：可视化文本或字符串数据

```python
import tensorflow as tf

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_float('lr', 0.5, 'learning rate')
tf.app.flags.DEFINE_integer('n_epoch', 100, 'num of epochs')
tf.app.flags.DEFINE_integer('n_feature', 13, 'num of features')
tf.app.flags.DEFINE_integer('n_hidden', 100, 'num of hidden units')


class Model:
    def __init__(self, sess, FLAGS):
        # 参数初始化
        self.sess = sess
        self.lr = FLAGS.lr
        self.n_epoch = FLAGS.n_epoch
        self.n_feature = FLAGS.n_feature
        self.n_hidden = FLAGS.n_hidden

        # 模型构建
        with tf.name_scope('Input'):
            self.X = tf.placeholder(tf.float32, [None, self.n_feature])
            self.Y = tf.placeholder(tf.float32, [None])

        with tf.name_scope('Variable'):
            self.W11 = tf.get_variable(
                name='W_11',
                shape=[self.n_feature, self.n_hidden],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.W12 = tf.get_variable(
                name='W_12',
                shape=[self.n_feature, self.n_hidden],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.W2 = tf.get_variable(
                name='W_2',
                shape=[self.n_hidden, 1],
                initializer=tf.contrib.layers.xavier_initializer()
            )
            self.B2 = tf.get_variable(
                name='B_2',
                shape=[1],
                initializer=tf.zeros_initializer()
            )

        with tf.name_scope('L11'):
            self.H11 = tf.nn.relu(tf.matmul(self.X, self.W11))

        with tf.name_scope('L12'):
            self.H12 = tf.nn.relu(tf.matmul(self.X, self.W12))

        with tf.name_scope('L2'):
            self.H2 = tf.matmul((self.H11 + self.H12), self.W2) + self.B2

        with tf.name_scope('Output'):
            self.predict = tf.squeeze(self.H2, -1)
            self.loss = tf.reduce_mean(tf.square(self.predict - self.Y))
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

        # 损失常量展示
        self.loss_summary = tf.summary.scalar('loss', self.loss)
        # 数据分布展示
        self.data_summary = tf.summary.histogram('data', self.X)
        # 权重分布展示
        self.weight = tf.summary.histogram('weight', self.W2)
        # 把所有 summary 合并起来
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter("logs/test", graph=self.sess.graph)
        self.test_writer = tf.summary.FileWriter("logs/train", graph=self.sess.graph)

    def run(self, train_data, test_data):
        x_train, y_train = train_data
        x_test, y_test = test_data
        
        # 数据标准化
        mean = x_train.mean(axis=0)
        std = x_train.std(axis=0)
        x_train = (x_train - mean) / std
        x_test = (x_test - mean) / std
        
        # 开始训练
        self.sess.run(tf.global_variables_initializer())
        run_options = tf.RunOptions(trace_level=tf.RunOptions.SOFTWARE_TRACE)
        run_metadata = tf.RunMetadata()
        for i in range(self.n_epoch):
            _, train_loss, train_summary = self.sess.run([self.optimizer, self.loss, self.summaries], feed_dict={self.X: x_train, self.Y: y_train},
                                     options=run_options, run_metadata=run_metadata)
            test_loss, test_summary = self.sess.run([self.loss, self.summaries], feed_dict={self.X: x_test, self.Y: y_test})
            # 打印图的软硬件使用信息
            self.train_writer.add_run_metadata(run_metadata, 'epoch %d' % i)
            self.train_writer.add_summary(train_summary, i)
            self.test_writer.add_summary(test_summary, i)
            print("Train loss:%s, Test loss:%s" % (train_loss, test_loss))


if __name__ == '__main__':
    train_data, test_data = tf.keras.datasets.boston_housing.load_data(
        path='boston_housing.npz',
        test_split=0.2,
        seed=113
    )
    config = tf.ConfigProto()
    config.gpu_options.per_process_gpu_memory_fraction = 0.01
    with tf.Session(config=tf.ConfigProto()) as sess:
        model = Model(sess, FLAGS)
        model.run(train_data, test_data)
```

