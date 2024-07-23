require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV.js');
const RedNeuronal = require('./red_neuronal.js'); // Cargamos la clase encargada de realizar la regresión lineal
const pathCSV = '../csv/';

//cargamos csv 
const { features , labels, testFeatures, testLabels } = loadCSV(
    `${pathCSV}datos-marzo-horas.csv`,
    ',',
    {
        shuffle: true,
        splitTest: 9,
        dataColumns: ['dia'],
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
});*/



console.log(models);
const redneuronalPrueba = new RedNeuronal(
    features, 
    labels,
    {
        learningRate: 0.1,
        epochs: 2000,
        neurons: 60,
    }
);

redneuronalPrueba.compilar();
redneuronalPrueba.entrenar();
const resultado = redneuronalPrueba.prediccion(labels);

console.log("Resultados obtenidos:");
resultado.print();
console.log("Resultados reales:");
console.log(labels);


