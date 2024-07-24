const tf = require('@tensorflow/tfjs'); // TensorFlow

//Calse encargada de toda la lógia de un algoritmo hecho por redes neuronales
class RedNeuronal{

    //Features = Datos , Labels = Resultados, Options = Opciones del algoritmo 
    constructor(features, labels, options){

        //Declaramos los valores independientes (features) y los valores dependientes (labels)
        this.features = tf.tensor(features);
        this.labels   = tf.tensor(labels);

        //Declaramos el modelo del algoritmo
        this.model = tf.sequential();

        //Declaramos las opciones
        this.options = Object.assign({ epochs: 2000, learningRate: 0.1, neurons: 10, activation: 'relu' }, options);
    };

    //Método que se encarga de compilar el algoritmo de red neuronal
    compilar(){

        //Se crean las capas que componen la red neuronal y se añaden al modelo
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
    
        //Compilamos nuestro modelo con los parámetros de optimización y de pérdida de error
        //Tipo de optimizadores => 'adam' - 'adagrad' - 'adamax' - 'rmsprop'
        //Tipo de pérdida       => 'huberLoss' - 'absoluteDifference' - 'meanSquareError'
        this.model.compile({
            optimizer: tf.train.adam(this.options.learningRate),
            loss: tf.losses.absoluteDifference
        });
    
    };

    //Método que se encarga de entrenar el algoritmo con los features y labels que se le pasan en el contructor
    async entrenar(){
        console.log('Entrenando');
        return await this.model.fit(this.features, this.labels, { epochs: this.options.epochs, verbose: 0 });
    };

    //Método que predice y devuelvo unos resultados a partir de los datos de entrada que se le propocionen(features)
    prediccion(features){
        features = tf.tensor(features)
        return this.model.predict( features );
    }
}

//Exoprtamos la clase para poder usarla desde fuera
module.exports = RedNeuronal;