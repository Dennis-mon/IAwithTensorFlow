const tf = require('@tensorflow/tfjs'); // TensorFlow
const _ = require('lodash');            // Loadash

//Calse encargada de toda la lógia de regresión lineal
class RegresionLineal{

    //Features = Datos , Labels = Resultados, Options = Opciones del algoritmo 
    constructor(features, labels, options){

        //Declaramos los valores independientes (features) y los valores dependientes (labels)
        this.features = this.processFeatures(features);
        this.labels = tf.tensor(labels);
        this.mseHistory = [];

        //Declaramos las opciones
        this.options = Object.assign({ learningRate: 0.1, iterations: 1000 }, options);

        // Declaramos los valores de 'm' y 'b' que queremos encontrar para nuestra ecuación
        //b - indice 0
        //m - indice 1
        this.weights = tf.zeros([this.features.shape[1],1])
    }

    //Entrenamiento de la IA el número de veces que tenga la variable 'iterations' asociado
    train(){
        for(let i = 0; i < this.options.iterations; i++){
            this.gradientDescent();
            this.recordMSE();
            this.updateLearningRate();
        }
    }

    //Aplicamos el gradiente descendiente
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

    //Probamos nuestro algoritmo para ver su eficacia
    test(testFeatures, testLabels){
        //Convertimos los parámtros en tensores
        testFeatures = this.processFeatures(testFeatures);
        testLabels = tf.tensor(testLabels);

        //Generamos las predicciones para nuestro algoritmo entrenado
        const preditions = testFeatures.matMul(this.weights);

        //Diferencia entre el resultado real y nuestras predicciones
        const res = testLabels.sub(preditions)
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
    }

    //Procesamos los datos de las features
    processFeatures(features){
        //Convertimos features en un tensor
        features = tf.tensor(features);
        
        //Normalizamos los valores del tensor
        if(this.mean && this.variance){
            features = features.sub(this.mean).div(this.variance.pow(0.5));
        } else{
            features = this.standardize(features);
        }

        //Concatena a nuestras features una columna de todo 1
        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        return features;
    }

    //Normalizamos
    standardize(features){
        //Obtenemos la media y la diferencia de nuestro tensor de features(datos)
        const { mean, variance } = tf.moments(features,0);

        this.mean = mean;
        this.variance = variance;

        //Normalizamos el tensor y los devolvemos
        return features.sub(mean).div(variance.pow(0.5));
    }

    //Guardamos el minimum square error
    recordMSE(){
        //Calculamos el minimum square error
        const mse = this.features
            .matMul(this.weights)
            .sub(this.labels)
            .pow(2)
            .sum()
            .div(this.features.shape[0])
            .get();

        //Guardamos el resultado en la primera posicion del array
        this.mseHistory.unshift(mse);
    }

    //Actualizamos el learning rate
    updateLearningRate(){
        if(this.mseHistory.length < 2){
            return;
        }

        // Si el minimum square error aumenta dividimos el learning rate por 2
        // sino, enotnces multiplicamos el learning rate por 0.05
        if(this.mseHistory[0] > this.mseHistory[1]){
            this.options.learningRate /= 2;
        } else {
            this.options.learningRate *= 1.05;
        }
    }

    //Predeccimos un resultado en concreto
    predictResult(features){
        features = tf.tensor(features);
        features = tf.reshape(features, [1, features.shape[0]]);
        features = features.sub(this.mean).div(this.variance.pow(0.5));
        features = tf.ones([features.shape[0], 1]).concat(features, 1);

        const preditions = features.matMul(this.weights);
        return preditions.sum().get();
    }

    //Mostramos el valor de b y m que hemos calculado
    mostrarPesos(){
        console.log('==========================')
        console.log('Mostrar Pesos')
        for(let i = 0; i < this.weights.shape[0]; i++){
            if(i == 0) console.log('Valor de B =>', this.weights.get(0,0));
            else console.log(`Valor de M${i} =>`, this.weights.get(i,0));
        }
        console.log('==========================')
    }
}


//Exoprtamos la clase para poder usarla desde fuera
module.exports = RegresionLineal;