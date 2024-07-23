const tf = require('@tensorflow/tfjs'); // TensorFlow


class RedNeuronal{

    constructor(features, labels, options){
        this.features = tf.tensor(features);
        this.labels   = tf.tensor(labels);

        this.model = tf.sequential();

        this.options = Object.assign({ epochs: 2000, learningRate: 0.1, neurons: 10, activation: 'relu' }, options);
        console.log(this.options.epochs);
    };

    compilar(){

        const hiddenLayer = tf.layers.dense({
            units: this.options.neurons,
            inputShape: [ this.features.shape[1] ],
            activation: this.options.activation
        });

        this.model.add(hiddenLayer);

        const outputLayer = tf.layers.dense({
            units: 1,
            inputShape: [this.options.neurons],
        });
    
        this.model.add(outputLayer);
    
        this.model.compile({
            optimizer: tf.train.adam(this.options.learningRate),
            loss: 'meanSquaredError'
        });
    
    };

    async entrenar(){
        console.log('Entrenando');
        console.log(this.options.epochs);
        await this.model.fit(this.features, this.labels, { epochs: this.options.epochs });
    };

    prediccion(inputs){
        inputs = tf.tensor(inputs)
        return this.model.predict( inputs );
    }
}

//Exoprtamos la clase para poder usarla desde fuera
module.exports = RedNeuronal;