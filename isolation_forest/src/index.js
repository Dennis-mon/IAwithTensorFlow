const IsolationForest = require('isolation-forest')
const loadCSVPercentage = require('./loadCSVPercentage')

const pathCSV = '../csv/';

// Colores ANSI
const reset = "\x1b[0m";
const green = "\x1b[32m";


//Cargamos CSV 
const { features , labels, testFeatures, testLabels } = loadCSVPercentage(
    `${pathCSV}energia-enero-7dias.csv`,
    ',',
    {
        shuffle: false,
        percentageTest: 20,
        dataColumns: ['dia de la semana', '0-24','value']
    }
)

//creamos isolation forest
var isolationForest = new IsolationForest.IsolationForest();
isolationForest.fit(features) // Type ObjectArray ({}[]); 

var trainingScores = isolationForest.scores()

//predecimos valores con las features
var scores = isolationForest.predict(testFeatures)

//creamos array donde colocaremos objetos con el indice y valor de anomalia de los elementos raros.
var anomalies = [];

//filtramos los anomalyScore mayores a 0.5
scores.forEach( (element, index) =>{
    if(element > 0.5) anomalies.push( {"anomalyScore":element, "index": index} );
})

//ordenamos de mayor a menos valor de anomalia
anomalies.sort((a, b) => b.anomalyScore - a.anomalyScore);

//mostramos resultados
anomalies.forEach( element => {
    console.log( `${green}${element.anomalyScore}${reset}: ` + testFeatures[element.index]);
});