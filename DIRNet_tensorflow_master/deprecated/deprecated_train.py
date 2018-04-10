def deprecated_main():
    # get configure
    config = get_config(is_train=True)

    # file operations
    if not os.path.exists(config["temp_dir"]):
        os.mkdir(config["temp_dir"])
    if not os.path.exists(config["checkpoint_dir"]):
        os.mkdir(config["checkpoint_dir"])

    #
    sess = tf.Session()
    reg = DIRNet(sess, config, "DIRNet", is_train=True)
    dh = MNISTDataHandler("MNIST_data", is_train=True)

    for i in range(config["iteration_num"]):
        batch_x, batch_y = dh.sample_pair(config["batch_size"])
        loss = reg.fit(batch_x, batch_y)
        print("iter {:>6d} : {}".format(i + 1, loss))

        if (i + 1) % 1000 == 0:
            reg.deploy(config["temp_dir"], batch_x, batch_y)
            reg.save(config["checkpoint_dir"])
