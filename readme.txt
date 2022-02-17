step1-pre 为预处理的程序：输出为train以及 test的h5格式的文件
step2-train为训练模型的程序：输出为训练的模型参数保存在checkpoint文件夹中
step3-test为预测的程序：输入为一张图片的h5文件，输出pnsr值。并且将input图片、label图片、预测图片保存在sample中够查看。

checkpoint文件夹：保存的模型参数
h5文件夹：存储的train、test训练集的h5文件
sample文件夹：保存的是input、label、预测图片

解释：
这里模型参数已经训练并保存了，直接运行strp-test加载保存在checkpoint文件夹中参数，最后可得到pnsr值。
输入为test文件夹中set文件夹中的一张图片（本来有五张，取一张先做实验），这里取得是woman图片。
得到的pnsr为29.09db，优于双三次插值的28.56db，但距离理想效果30.92db仍有距离。
通过肉眼对sample文件夹中的图片观察，很明显预测的图片相比input和label都偏黑，不知道是不是原因在这里。