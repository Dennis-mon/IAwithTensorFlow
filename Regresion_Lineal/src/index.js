require('@tensorflow/tfjs-node');
const tf = require('@tensorflow/tfjs');
const loadCSV = require('./loadCSV');

const {features, labels, testFeatures, testLabels} = loadCSV(
    './src/csv/datos.csv',
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
console.log("testLabels: ", testLabels );