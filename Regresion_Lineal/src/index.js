require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV');
const RegresionLineal = require('./regresion_lineal'); // Cargamos la clase encargada de realizar la regresión lineal

const pathCSV = './src/csv/';

/*const {features, labels, testFeatures, testLabels} = loadCSV(
    `${pathCSV}datos.csv`,
    ',', 
    {
        dataColumns: ['mes','dia'],
        labelColumns: ['value'],
        shuffle: true,
        splitTest: false,
        converters:{}
    }
)

console.log("features: ", features );
console.log("labels: ", labels );
console.log("testFeatures: ", testFeatures );
console.log("testLabels: ", testLabels );*/


const { features , labels, testFeatures, testLabels } = loadCSV(`${pathCSV}cars.csv`, ',', {
    shuffle: true,
    splitTest: 50,
    dataColumns: ['horsepower', 'weight', 'displacement'],
    labelColumns: ['mpg']
})

const regresionPrueba = new RegresionLineal(features, labels, {
    learningRate: 10,
    iterations: 100
});

//regresionPrueba.features.print();

regresionPrueba.train();
const r = regresionPrueba.test(testFeatures, testLabels);

console.log('Valor de B =>', regresionPrueba.weights.get(0,0));
console.log('Valor de M1 =>', regresionPrueba.weights.get(1,0));
console.log('Valor de M2 =>', regresionPrueba.weights.get(2,0));
console.log('Valor de M3 =>', regresionPrueba.weights.get(3,0));
console.log('R => ', r);

console.log('==========================')
console.log('Prueba');
console.log('Datos =>' , features[1]);
console.log('Resultado =>' , labels[1]);
regresionPrueba.predictResult(features[1]);