import pickle
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm
from stride import StridePattern, initialize_network
from conversion import get_clip_ops, update_net_from_tf, Cost

from tensorflow.python.framework.ops import disable_eager_execution

disable_eager_execution()


def compare_stride_patterns(system, n=5):
    """
    Creates and optimizes networks with several random stride patterns, to
    explore sensitivity of optimization to stride pattern.

    :param system: a System (physiological model) for which to create a convnet
    :param n: number of stride patterns / networks to generate
    """
    import matplotlib.pyplot as plt

    nets = [None] * n
    training_curves = [None] * n
    for i in range(n):
        tf.reset_default_graph()
        nets[i], training_curves[i] = optimize_network_architecture(system)
        tc = np.array(training_curves[i])
        print(tc.shape)
        plt.semilogy(tc[:, 0], tc[:, 1])

    data = {"training_curves": training_curves, "nets": nets}
    with open("nets-and-training-curves.pkl", "wb") as f:
        pickle.dump(data, f)

    plt.show()


def optimize_network_architecture(system, stride_pattern=None, compare=True):
    """
    :param system: a System (physiological model) to fit network architecture to
    :param stride_pattern: strides for each connection (generated at random if not given
    :param compare: print comparison of optimized values with target values
    :return: optimized network, training curve
    """

    if not stride_pattern:
        stride_pattern = StridePattern(system, 32)
        stride_pattern.fill()

    net = initialize_network(system, stride_pattern, image_layer=0, image_channels=3.0)

    optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=0.00001)
    print("Setting up cost structure")
    cost = Cost(system, net)

    print("Defining cost function")

    # pc = cost.param_cost(1e-13)
    fc = cost.match_cost_f(1.0)
    bc = cost.match_cost_b(1.0)
    ec = cost.match_cost_e(1.0)
    wc = cost.match_cost_w(1.0)
    dec = cost.dead_end_cost(1.0)
    kc = cost.w_k_constraint_cost(1.0)
    rfc = cost.w_rf_constraint_cost(1.0)

    c = fc + bc + ec + wc + dec + 5 * kc + 5 * rfc

    vars = [cost.network.c, cost.network.sigma]
    for w_rf in cost.network.w_rf:
        if isinstance(w_rf, tf.Variable):
            vars.append(w_rf)

    clip_ops = [get_clip_ops(cost.network.c), get_clip_ops(cost.network.sigma)]

    # Optimiser, and global_step variables on the CPU
    global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name="global_step")

    print("Setting up optimizer")
    opt_op = optimizer.minimize(c, global_step=global_step, var_list=vars)

    # Create our model saver to save all the trainable variables and the global_step
    save_vars = {v.name: v for v in tf.compat.v1.trainable_variables()}
    save_vars.update({global_step.name: global_step})
    saver = tf.compat.v1.train.Saver(save_vars)

    # Create our summary operation
    summary_op = tf.compat.v1.summary.merge(
        [
            tf.compat.v1.summary.scalar("f", fc),
            tf.compat.v1.summary.scalar("b", bc),
            tf.compat.v1.summary.scalar("e", ec),
            tf.compat.v1.summary.scalar("w", wc),
            tf.compat.v1.summary.scalar("de", dec),
            tf.compat.v1.summary.scalar("k", kc),
            tf.compat.v1.summary.scalar("rf", rfc),
            tf.compat.v1.summary.scalar("c", c),
        ]
    )

    init = tf.compat.v1.global_variables_initializer()
    with tf.compat.v1.Session() as sess:
        print("Initializing")
        sess.run(init)
        print("Printing")
        update_net_from_tf(sess, net, cost.network)
        net.print()

        # Setup for tensorboard
        with tf.compat.v1.summary.FileWriter("logs", graph=tf.compat.v1.get_default_graph()) as summary_writer:
            training_curve = []
            c_value, summary = sess.run([c, summary_op])
            training_curve.append((0, c_value, sess.run(dec), sess.run(kc)))
            _print_cost(c_value, sess.run(fc), sess.run(bc), sess.run(ec), sess.run(wc), sess.run(dec))

            # Write out to the summary
            summary_writer.add_summary(summary, tf.compat.v1.train.global_step(sess, global_step))

            iterations = 100
            for i in range(2001):
                start = time.perf_counter()
                _run_optimization_steps(sess, opt_op, iterations=iterations, clip_ops=clip_ops)
                cost_i, summary = sess.run([c, summary_op])
                end = time.perf_counter()

                training_curve.append((iterations * i, cost_i, sess.run(dec), sess.run(kc)))
                print(
                    "Batch: {} ({:2f}s) Cost: {:3g}".format(
                        tf.compat.v1.train.global_step(sess, global_step), (end - start), cost_i
                    )
                )

                # Write out to the summary
                summary_writer.add_summary(summary, tf.compat.v1.train.global_step(sess, global_step))

                if np.isnan(cost_i):
                    _print_cost(cost_i, sess.run(fc), sess.run(bc), sess.run(ec), sess.run(wc), sess.run(dec))
                    break

                if i > 0 and i % 50 == 0:
                    print("saving checkpoint")
                    saver.save(sess, "outputs/msh", tf.compat.v1.train.global_step(sess, global_step))
                    update_net_from_tf(sess, net, cost.network)
                    with open("outputs/optimization-checkpoint.pkl", "wb") as file:
                        pickle.dump({"net": net, "training_curve": training_curve}, file)

            saver.save(sess, "outputs/msh", tf.compat.v1.train.global_step(sess, global_step))
            update_net_from_tf(sess, net, cost.network)
            with open("outputs/optimization-checkpoint.pkl", "wb") as file:
                pickle.dump({"net": net, "training_curve": training_curve}, file)
            net.print()

            if compare:
                cost.compare_system(system, sess)

    return net, training_curve


def _run_optimization_steps(sess, opt_op, iterations=100, clip_ops=[]):
    for _ in tqdm(range(iterations)):
        opt_op.run()
        for clip_op in clip_ops:
            sess.run(clip_op)


def _print_cost(total_cost, f_cost, b_cost, e_cost, rf_cost, de_cost):
    print(
        "total cost: {} f cost: {} b cost {} e cost {} RF cost: {} dead-end cost: {}".format(
            total_cost, f_cost, b_cost, e_cost, rf_cost, de_cost
        )
    )
