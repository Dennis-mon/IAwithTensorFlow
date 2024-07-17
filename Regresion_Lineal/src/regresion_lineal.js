const tf = require('@tensorflow/tfjs'); // TensorFlow
const _ = require('lodash');            // Loadash

//Calse encargada de toda la lógia de regresión lineal
class RegresionLineal{

    // Features = , Labels = , Options = 
    constructor(features, labels, options){

        //Declaramos los valores independientes (features) y los valores dependientes (labels)
        this.features = tf.tensor(features);
        this.labels = tf.tensor(labels);

        //concatena a nuestras featuers una columna de todo 1
        this.features = tf.ones( [this.features.shape[0],1] ).concat(this.features,1);

        //Declaramos las opciones
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);

        // Declaramos los valores de 'm' y 'b' que queremos encontrar para nuestra ecuación
        //b - indice 0
        //m - indice 1
        this.weights = tf.zeros([2,1])
    }

    // Entrenamiento de la IA el número de veces que tenga la variable 'iterations' asociado
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
        //esta funcion es para multiplicacion entre matrices
        const currentGuess = this.features.matMul(this.weights);
        const differences = currentGuess.sub(this.labels);

        //calculo de pendiente de MSE respecto m y b
        const slopes = this.features
            .transpose()        //para poder multiplicar la diferencia por las features la transponemos
            .matMul(differences)
            .div(this.features.shape[0])

        //Actualizamos los valores de 'b' y 'm'
        this.weights = this.weights.sub(slopes.mul(this.options.learningRate));
    }
}


//Exoprtamos la clase para poder usarla desde fuera
module.exports = RegresionLineal;