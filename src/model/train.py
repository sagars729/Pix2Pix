import os
import sys
import time
import numpy as np
import models
from keras.utils import generic_utils
from keras.optimizers import Adam, SGD
import keras.backend as K
# Utils
#sys.path.append("../utils")
import general_utils
import data_utils

std_label = lambda x, f: x if not f else "_latest"

try: import cPickle as pickle
except: import pickle
    
import logging
import threading
import time
    
def l1_loss(y_true, y_pred):
    return K.sum(K.abs(y_pred - y_true), axis=-1)

def thread_analyze_data(full, sketch, numbatch, model, res, analyze):
    full, sketch = data_utils.gen_batch_random(full, sketch, numbatch)
    full, sketch, gen = data_utils.get_generated_batch(full, sketch, model)
    res[0] = analyze.analyze(full, gen)
    
def thread_analyze(full_train, sketch_train, full_val, sketch_val, numbatch, model, analyze, analysis, e):
    train = [0]
    x = threading.Thread(target=thread_analyze_data, args=(full_train, sketch_train, numbatch, model, train, analyze))
    x.start()
    
    val = [0]         
    y = threading.Thread(target=thread_analyze_data, args=(full_val, sketch_val, numbatch, model, val, analyze))
    y.start()
    
    x.join()
    y.join()
    analysis[analyze.__str__()][e] = (train[0], val[0])
    print(analyze.__str__(), "Analysis: Training - %d Validation - %d" % analysis[analyze.__str__()][e])

def train(**kwargs):
    """
    Train model

    Load the whole train data in memory for faster operations

    args: **kwargs (dict) keyword arguments that specify the model hyperparameters
    """

    # Roll out the parameters
    batch_size = kwargs["batch_size"]
    n_batch_per_epoch = kwargs["n_batch_per_epoch"]
    nb_epoch = kwargs["nb_epoch"]
    model_name = kwargs["model_name"]
    generator = kwargs["generator"]
    image_data_format = kwargs["image_data_format"]
    img_dim = kwargs["img_dim"]
    patch_size = kwargs["patch_size"]
    bn_mode = kwargs["bn_mode"]
    label_smoothing = kwargs["use_label_smoothing"]
    label_flipping = kwargs["label_flipping"]
    dset = kwargs["dset"]
    use_mbd = kwargs["use_mbd"]
    do_plot = kwargs["do_plot"]
    logging_dir = kwargs["logging_dir"]
    save_every_epoch = kwargs["epoch"]
    save_latest = kwargs["save_latest"]
    load_model = kwargs["load_model"]
    load_epoch = kwargs["load_epoch"]
    base_epoch = kwargs["base_epoch"]
    analyze = kwargs["analyze"]
    analyze_batch = kwargs["analyze_batch"]
    load_analysis = kwargs["load_analysis"]
    
    epoch_size = n_batch_per_epoch * batch_size

    # Setup environment (logging directory etc)
    general_utils.setup_logging(model_name, logging_dir=logging_dir)

    # Load and rescale data
    X_full_train, X_sketch_train, X_full_val, X_sketch_val = data_utils.load_data(dset, image_data_format, logging_dir)
    img_dim = X_full_train.shape[-3:]

    # Get the number of non overlapping patch and the size of input image to the discriminator
    nb_patch, img_dim_disc = data_utils.get_nb_patch(img_dim, patch_size, image_data_format)
    
    
    analysis = {"disc_loss": {}, 
                "gen_loss": {}, 
                analyze.__str__(): {}}
    
    if load_analysis:  
        try:
            infile = open(os.path.join(logging_dir, "figures", model_name, load_analysis), "rb")
            analysis = pickle.load(infile)
            infile.close()
        except: print("Loading Analysis Failed")

    try:

        # Create optimizers
        opt_dcgan = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        # opt_discriminator = SGD(lr=1E-3, momentum=0.9, nesterov=True)
        opt_discriminator = Adam(lr=1E-3, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

        # Load generator model
        generator_model = models.load("generator_unet_%s" % generator,
                                      img_dim,
                                      nb_patch,
                                      bn_mode,
                                      use_mbd,
                                      batch_size,
                                      do_plot,
                                      load_model,
                                      logging_dir,
                                      load_epoch)
        # Load discriminator model
        discriminator_model = models.load("DCGAN_discriminator",
                                          img_dim_disc,
                                          nb_patch,
                                          bn_mode,
                                          use_mbd,
                                          batch_size,
                                          do_plot,
                                          load_model,
                                          logging_dir,
                                          load_epoch)

        generator_model.compile(loss='mae', optimizer=opt_discriminator)
        discriminator_model.trainable = False

        DCGAN_model = models.DCGAN(generator_model,
                                   discriminator_model,
                                   img_dim,
                                   patch_size,
                                   image_data_format,
                                   load_model,
                                   logging_dir,
                                   load_epoch)

        loss = [l1_loss, 'binary_crossentropy']
        loss_weights = [1E1, 1]
        DCGAN_model.compile(loss=loss, loss_weights=loss_weights, optimizer=opt_dcgan)

        discriminator_model.trainable = True
        discriminator_model.compile(loss='binary_crossentropy', optimizer=opt_discriminator)

        gen_loss = 100
        disc_loss = 100

        # Start training
        print("Start training")
        for e in range(base_epoch, nb_epoch):
            # Initialize progbar and batch counter
            progbar = generic_utils.Progbar(epoch_size)
            batch_counter = 1
            start = time.time()

            for X_full_batch, X_sketch_batch in data_utils.gen_batch(X_full_train, X_sketch_train, batch_size):

                # Create a batch to feed the discriminator model
                X_disc, y_disc = data_utils.get_disc_batch(X_full_batch,
                                                           X_sketch_batch,
                                                           generator_model,
                                                           batch_counter,
                                                           patch_size,
                                                           image_data_format,
                                                           label_smoothing=label_smoothing,
                                                           label_flipping=label_flipping)

                # Update the discriminator
                disc_loss = discriminator_model.train_on_batch(X_disc, y_disc)

                # Create a batch to feed the generator model
                X_gen_target, X_gen = next(data_utils.gen_batch(X_full_train, X_sketch_train, batch_size))
                y_gen = np.zeros((X_gen.shape[0], 2), dtype=np.uint8)
                y_gen[:, 1] = 1

                # Freeze the discriminator
                discriminator_model.trainable = False
                gen_loss = DCGAN_model.train_on_batch(X_gen, [X_gen_target, y_gen])
                # Unfreeze the discriminator
                discriminator_model.trainable = True

                batch_counter += 1
                progbar.add(batch_size, values=[("D logloss", disc_loss),
                                                ("G tot", gen_loss[0]),
                                                ("G L1", gen_loss[1]),
                                                ("G logloss", gen_loss[2])])

                # Save images for visualization
                if batch_counter % (n_batch_per_epoch / 2) == 0:
                    # Get new images from validation
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                    batch_size, image_data_format, "training",
                                                    logging_dir, model_name, e)
                    X_full_batch, X_sketch_batch = next(data_utils.gen_batch(X_full_val, X_sketch_val, batch_size))
                    data_utils.plot_generated_batch(X_full_batch, X_sketch_batch, generator_model,
                                                    batch_size, image_data_format, "validation",
                                                    logging_dir, model_name, e)

                if batch_counter >= n_batch_per_epoch:
                    break

            print("")
            print('Epoch %s/%s, Time: %s' % (e + 1, nb_epoch, time.time() - start))

            analysis["gen_loss"][e] = gen_loss[2]
            analysis["disc_loss"][e] = disc_loss
            
            if e % save_every_epoch == 0:
                gen_weights_path = os.path.join(logging_dir, 'models/%s/gen_weights_epoch%s.h5' % (model_name, std_label(e,save_latest)))
                generator_model.save_weights(gen_weights_path, overwrite=True)

                disc_weights_path = os.path.join(logging_dir, 'models/%s/disc_weights_epoch%s.h5' % (model_name, std_label(e,save_latest)))
                discriminator_model.save_weights(disc_weights_path, overwrite=True)

                DCGAN_weights_path = os.path.join(logging_dir, 'models/%s/DCGAN_weights_epoch%s.h5' % (model_name, std_label(e,save_latest)))
                DCGAN_model.save_weights(DCGAN_weights_path, overwrite=True)
                
                if analyze: 
                    #t = threading.Thread(target=thread_analyze, args=(X_full_train, X_sketch_train, X_full_val, X_sketch_val, analyze_batch, generator_model, analyze, analysis, e))
                    #t.start()
                    X_full_batch, X_sketch_batch = data_utils.gen_batch_random(X_full_train, X_sketch_train, analyze_batch)
                    X_full_batch, X_sketch_batch, X_gen_batch = data_utils.get_generated_batch(X_full_batch, X_sketch_batch, generator_model)
                    train_analysis = analyze.analyze(X_full_batch, X_gen_batch)
                    
                    X_full_batch, X_sketch_batch = data_utils.gen_batch_random(X_full_val, X_sketch_val, analyze_batch)
                    X_full_batch, X_sketch_batch, X_gen_batch = data_utils.get_generated_batch(X_full_batch, X_sketch_batch, generator_model)
                    val_analysis = analyze.analyze(X_full_batch, X_gen_batch)
                    
                    analysis[analyze.__str__()][e] = (train_analysis, val_analysis)
                    print(analyze.__str__(), "Analysis: Training - %d Validation - %d" % analysis[analyze.__str__()][e])
                    
                    if load_analysis:
                        try:
                            outfile = open(os.path.join(logging_dir, "figures", model_name, load_analysis), "wb")
                            pickle.dump(analysis,outfile)
                            outfile.close()
                        except: print("Saving Analysis Failed")
                
                print('Time: %s' % (time.time() - start))
                    
    except KeyboardInterrupt:
        pass
    
    return analysis
