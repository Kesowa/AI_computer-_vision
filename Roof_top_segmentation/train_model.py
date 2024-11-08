model = unet.get_model(input_shape=(224, 224, 3), output_channels=1, output_act=None)

util.show_predictions(train, model, num=3)

epochs = 50
steps_per_epoch = nb_train // batch_size
val_steps = nb_val // batch_size

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)
model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

callbacks = [util.DisplayCallback(train, model, intv=10)]
history = model.fit(train,
                    epochs=epochs,
                    steps_per_epoch=steps_per_epoch,                              
                    validation_data=val,
                    validation_steps=val_steps,
                    callbacks=callbacks)
# C:\Users\FS-AI\Desktop\BACKUP\work\data\NAR\RS\rts-tf2\data.py
