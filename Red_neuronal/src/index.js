require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV.js');
const RedNeuronal = require('./red_neuronal.js'); // Cargamos la clase encargada de realizar la regresión lineal
const pathCSV = '../csv/';

//Cargamos CSV 
const { features , labels, testFeatures, testLabels } = loadCSV(
    `${pathCSV}datos-marzo-horas.csv`,
    ',',
    {
        shuffle: true,
        splitTest: 9,
        dataColumns: ['dia', 'mes', '0-24'],
        labelColumns: ['value']
    }
)

/*
function Model( model, activation, error ){
    this.model = model;
    this.activation = activation;
    this.error = error;
}

let model1 = new Model("modelo1", "relu", 0.2);
let model2 = new Model("modelo2", "sigmoid", 0.1);
let model3 = new Model("modelo3", "tanh", 100);

let models = [];
models.push(model1);
models.push(model2);
models.push(model3);

// Ordenar el array por el valor de error en orden ascendente
models.sort(function(a, b) {
    return a.error - b.error;
});

console.log(models);

*/


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
        learningRate: 0.1,
        epochs: 2000,
        neurons: 32,
    }
);

//Función asincrona que se encarga de compilar y entrenar el algorimto ademas de generar una prediccón para unos valores independientes de entrada(features)
async function crearRedes(){

    redneuronalPrueba.compilar();
    await redneuronalPrueba.entrenar(); //await para impedir que el código se siga ejecutando hasta que el algoritmo este 100% entrenado

    //Guardamos los resultados de la predicción en una variable auxilar para poder verlos en la terminal
    const resultado = redneuronalPrueba.prediccion(features);
    console.log("Resultados obtenidos:");
    resultado.print();
    console.log("Resultados reales:");
    console.log(labels);
}

crearRedes();
