const tf = require('@tensorflow/tfjs'); // TensorFlow
const _ = require('lodash');            // Loadash

//Calse encargada de toda la lógia de regresión lineal
class RegresionLineal{

    //Features = , Labels = , Options = 
    constructor(features, labels, options){

        //Declaramos los valores independientes (features) y los valores dependientes (labels)
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);

        //Declaramos las opciones
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);

        // Declaramos los valores de 'm' y 'b' que queremos encontrar para nuestra ecuación
        //b - indice 0
        //m - indice 1
        this.weights = tf.zeros([2,1])
    }

    //Entrenamiento de la IA el número de veces que tenga la variable 'iterations' asociado
    train(){
        for(let i = 0; i < this.options.iterations; i++){
            this.gradientDescent();
        }
    }

    //Ajustamos los valores de 'm' y 'b' a su resultado óptimo
    /*gradientDescent(){
        //Generamos unas predicciones con los valores actuales de 'b' y 'm'
        const currentGuess = this.features.map( row => {
            return this.m * row[0] + this.b;
        });

        //Calculamos los slope de 'b' y 'm' respectivamente
        const bSlope = _.sum(currentGuess.map((guess, i) => {
            return guess - this.labels[i][0];
        })) * 2 / this.labels.length;

        const mSlope = _.sum(currentGuess.map((guess, i) => {
            return -1*this.features[i][0] * (this.labels[i][0] - guess)
        })) * 2 / this.labels.length;

        //Actualizamos los valores de 'b' y 'm'
        this.m = this.m - mSlope * this.options.learningRate;
        this.b = this.b - bSlope * this.options.learningRate;
    }*/

    gradientDescent(){
        //Esta funcion es para multiplicacion entre matrices
        const currentGuess = this.features.matMul(this.weights);
        const differences = currentGuess.sub(this.labels);

        //Calculo de pendiente de MSE respecto m y b
        const slopes = this.features
            .transpose()        //para poder multiplicar la diferencia por las features la transponemos
            .matMul(differences)
            .div(this.features.shape[0])

        //Actualizamos los valores de 'b' y 'm'
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }

    test(testFeatures, testLabels){
        //Convertimos los parámtros en tensores
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        //Generamos las predicciones para nuestro algoritmo entrenado
        const preditions = testFeatures.matMul(this.weights);

        const res = testLabels.sub(preditions)
            .pow(2)
            .sum()
            .get();

        const tot = testLabels.sub(testLabels.mean())
            .pow(2)
            .sum()
            .get();

        return 1 - res / tot;
    }

    processFeatures(features){
        //Convertimos features en un tensor
        features = tf.tensor(features);
        
        if(this.mean && this.variance){
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else{
            features = this.standardize(features);
        }
        
        //Concatena a nuestras featuers una columna de todo 1
        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    standardize(features){
        const { mean, variance } = tf.moments(features,0);

        this.mean = mean;
        this.variance = variance;

        return features.sub(mean).div(variance.pow(0.5));
    }
}


//Exoprtamos la clase para poder usarla desde fuera
module.exports = RegresionLineal;