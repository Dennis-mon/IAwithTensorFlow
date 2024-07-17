const tf = require('@tensorflow/tfjs'); // TensorFlow
const _ = require('lodash');            // Loadash

//Calse encargada de toda la lógia de regresión lineal
class RegresionLineal{

    // Features = , Labels = , Options = 
    constructor(features, labels, options){

        //Declaramos ...
        this.features = features;
        this.labels = labels;

        //Declaramos las opciones
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);

        // Declaramos los valores de 'm' y 'b' que queremos encontrar para nuestra ecuación
        this.m = 0;
        this.b = 0;
    }

    // Entrenamiento de la IA el número de veces que tenga la variable 'iterations' asociado
    train(){
        for(let i = 0; i < this.options.iterations; i++){
            this.gradientDescent();
        }
    }

    //Ajustamos los valores de 'm' y 'b' a su resultado óptimo
    gradientDescent(){
        //Generamos unas predicciones con los valores actuales de 'b' y 'm'
        const currentGuess = this.features.map( row => {
            return this.m * row[0] + this.b;
        });

        //Calculamos los scope de 'b' y 'm' respectivamente
        const bSlope = _.sum(currentGuess.map((guess, i) => {
            return guess - this.labels[i][0];
        })) * 2 / this.labels.length;

        const mSlope = _.sum(currentGuess.map((guess, i) => {
            return -1*this.features[i][0] * (this.labels[i][0] - guess)
        })) * 2 / this.labels.length;

        //Actualizamos los valores de 'b' y 'm'
        this.m = this.m - mSlope * this.options.learningRate;
        this.b = this.b - bSlope * this.options.learningRate;
    }
}


//Exoprtamos la clase para poder usarla desde fuera
module.exports = RegresionLineal;