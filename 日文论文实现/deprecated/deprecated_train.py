    # # R1, R2, R3 path
    # "train_in_x_dir_all": r"F:\registration_running_data\validate_3",
    # "train_in_y_dir_all": r"F:\registration_patches\version_all\train\resized_ct",
    # "valid_in_x_dir_all": r"F:\registration_running_data\validate_3",
    # "valid_in_y_dir_all": r"F:\registration_patches\version_all\train\resized_ct",
    # "valid_out_dir_all": r"F:\registration_running_data\validate_all",
    # # 再统一训练R1 + R2 + R3
    # train_x_all, train_y_all = gen_batches(config["train_in_x_dir_all"], config["train_in_y_dir_all"], {
    #     "batch_size": config["batch_size"],
    #     "image_size": config["image_size"],
    #     "shuffle_batch": True
    # })
    # valid_x_all, valid_y_all = gen_batches(config["valid_in_x_dir_all"], config["valid_in_y_dir_all"], {
    #     "batch_size": config["batch_size"],
    #     "image_size": config["image_size"],
    #     "shuffle_batch": False
    # })
    # coord_all = tf.train.Coordinator()
    # threads_all = tf.train.start_queue_runners(sess=sess, coord=coord_all)
    # for i in range(config["epoch_num"]):
    #     _tx_all, _ty_all = sess.run([train_x_all, train_y_all])
    #     loss = reg.fit(_tx_all, _ty_all)
    #     print("[INFO] epoch={:>5}, loss={:.3f}".format(i, loss))
    #     if (i + 1) % config["save_interval"] == 0:
    #         # reg.save(sess, config["checkpoint_dir"])
    #         pass
    # for j in range(valid_iter_num):
    #     _vx_all, _vy_all = sess.run([valid_x_all, valid_y_all])
    #     reg.deploy(config["valid_out_dir_all"], _vx_all, _vy_all, j * config["batch_size"])
    # coord_all.request_stop()
    # coord_all.join(threads_all)


    # if (i + 1) % config["save_interval"] == 0:
    #     # reg.save(sess, config["checkpoint_dir"])
    #     pass