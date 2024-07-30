require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSVPercentage = require('./loadCSVPercentage.js');
const RedNeuronal = require('./red_neuronal.js'); // Cargamos la clase encargada de realizar la regresión lineal
const plot = require('node-remote-plot'); // Pacakage para generar gráficos
const { forEach } = require('lodash');
const pathCSV = '../csv/';

//Cargamos CSV 
const { features , labels, testFeatures, testLabels } = loadCSVPercentage(
    `${pathCSV}energia-enero-7dias.csv`,
    ',',
    {
        shuffle: true,
        percentageTest: 0,
        dataColumns: [ 'dia de la semana', '0-24'],
        labelColumns: ['value']
    }
)

//PARA CREAR REDES NEURONALES UNA A UNA

//Creamos una nueva instancia de la clase Res Neuronal
const redneuronalPrueba = new RedNeuronal(
    features, //Variables Independientes
    labels,   //Variables Dependientes
    //Configuración
    //learningRate = tasa de aprendizaje // por defecto 0.1
    //epochs       = nº de veces que se entrena el algoritmo para todos los datos // por fecefto 2000 
    //neurons      = nº de neuronas que queramos que tenga el algoritmo // por defecto 10
    //activation   = función de activación que vamos a usar para el algoritmo, se recomienda ('relu' , 'tanh', 'sigmoid', 'lineal') // por defecto 'relu'
    {
        learningRate: 0.001,
        epochs: 50,
        neurons: 10,
        activation: 'relu',
        percentage_train: 80,
        batchsize: 50
    }
);

//Función asincrona que se encarga de compilar y entrenar el algorimto ademas de generar una prediccón para unos valores independientes de entrada(features)
async function crearRedes(){

    redneuronalPrueba.compilar();
    const historial = await redneuronalPrueba.entrenar(); //await para impedir que el código se siga ejecutando hasta que el algoritmo este 100% entrenado

    //Guardamos los resultados de la predicción en una variable auxilar para poder verlos en la terminal
    const resultado = redneuronalPrueba.prediccion(features);
    console.log("Resultados obtenidos:");
    resultado.print();
    console.log("Resultados reales:");
    console.log(labels);
    //console.log('Testeo');
    //console.log(redneuronalPrueba.testeo(testFeatures, testLabels));

    //Ploteamos 
    plot({   
        x: historial.history.loss,
        xLabel: 'Iteration #',
        yLabel: 'Mean Square Error',
        title: 'MSE',
        name: 'MSE'
    });

    /*plot({
        x: historial.history.acc,
        xLabel: 'Iteration #',
        yLabel: 'Accuracy',
        title: 'Accuracy',
        name: 'Accuracy'
    });*/
}

crearRedes();

//=======================================
