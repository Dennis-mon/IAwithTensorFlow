const tf = require('@tensorflow/tfjs'); // TensorFlow

//Calse encargada de toda la lógia de un algoritmo hecho por redes neuronales
class RedNeuronal2{

    //Features = Datos , Labels = Resultados, Options = Opciones del algoritmo 
    constructor(features, labels, options, inputLayer, hiddenLayers,outputLayer){

        //Declaramos los valores independientes (features) y los valores dependientes (labels)
        this.features = tf.tensor(features);
        this.labels   = tf.tensor(labels);

        //Declaramos el modelo del algoritmo
        this.model = tf.sequential();

        //Declaramos las opciones
        this.options = Object.assign({ epochs: 2000, learningRate: 0.1}, options);

        this.inputLayer = Object.assign({ neurons: 1, activation: 'relu'}, inputLayer);

        this.hiddenLayers = hiddenLayers;

        this.outputLayer = Object.assign({ neurons: 1, activation: 'relu'}, outputLayer);

    };

    //Método que se encarga de compilar el algoritmo de red neuronal
    compilar(){

        //Creamos capa input
        const inputLayer = tf.layers.dense({
            units: this.inputLayer.neurons,
            inputShape: [ this.features.shape[1] ],
            activation: this.inputLayer.activation
        });
        //la añadimos al modelo
        this.model.add(inputLayer);

        //creamos capas intermedias, y las vamos añadiendo
        this.hiddenLayers.forEach( layer => {

            const hiddenLayer = tf.layers.dense({
                units: layer.neurons,
                //inputShape: [ this.features.shape[1] ],
                activation: layer.activation
            });

            this.model.add(hiddenLayer);
        });

        
        //creamos capa de salida
        const outputLayer = tf.layers.dense({
            units: this.outputLayer.neurons,
            //inputShape: [ this.features.shape[1] ],
            activation: this.outputLayer.activation
        });
    
        this.model.add(outputLayer);
    
        //Compilamos nuestro modelo con los parámetros de optimización y de pérdida de error
        //Tipo de optimizadores => 'adam' - 'adagrad' - 'adamax' - 'rmsprop'
        //Tipo de pérdida       => 'huberLoss' - 'absoluteDifference' - 'meanSquareError'
        this.model.compile({
            optimizer: tf.train.adam(this.options.learningRate),
            loss: tf.losses.meanSquaredError
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
    };

    //Método para ver la eficacia del algoritmo entrenado
    testeo(testFeatures, testLabels){
        testFeatures = tf.tensor(testFeatures);
        testLabels   = tf.tensor(testLabels);

        const resultado = this.model.predict( testFeatures );

        //Diferencia entre el resultado real y nuestras predicciones
        const res = testLabels.sub(resultado)
            .pow(2)
            .sum()
            .get();
    
        //Diferencia entre el resultado real y la media de dichos datos
        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .get();
    
        //Para que el algoritmo sea funcional su resultado tiene que estar los mas cercano a 1 posible
        //Si el resultado del test de un número negativo significa que el algoritmo funciona tan mal que sería mejor hacer una media directamente
        return 1 - res / tot;
    };
}

//Exoprtamos la clase para poder usarla desde fuera
module.exports = RedNeuronal;