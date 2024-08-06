const IsolationForest = require('isolation-forest')
const loadCSVPercentage = require('./loadCSVPercentage')

const { createCanvas } = require('canvas');
const {Chart, registerables} = require('chart.js');
const fs = require('fs');
const { values } = require('lodash');

require('chartjs-adapter-moment');
const moment = require('moment');
// Registrar todos los componentes necesarios
Chart.register(...registerables);

const pathCSV = '../csv/';

// Colores ANSI
const reset = "\x1b[0m";
const green = "\x1b[32m";


//Cargamos CSV 
const { features , labels, testFeatures, testLabels } = loadCSVPercentage(
    `${pathCSV}prueba.csv`,
    ',',
    {
        shuffle: false,
        percentageTest: 20,
        dataColumns: ['dia de la semana', '0-24','value'],
        labelColumns: ['año','mes','dia','hora','minuto']
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
var normal = [];


//filtramos los anomalyScore mayores a 0.5
scores.forEach( (element, index) =>{
    if(element > 0.5){  
        
        let value = features[index][2];
        date = new Date( labels[index][0],labels[index][1],labels[index][2],labels[index][3],labels[index][4]);
        date = moment(date).format('YYYY-MM-DD HH:mm')
        anomalies.push({"valor": value, "date": date});
    } 
    else {
        let value = features[index][2];
        date = new Date( labels[index][0],labels[index][1],labels[index][2],labels[index][3],labels[index][4]);
        date = moment(date).format('YYYY-MM-DD HH:mm')
        normal.push({"valor": value, "date": date});
    }
})

//ordenamos de mayor a menos valor de anomalia
//anomalies.sort((a, b) => b.anomalyScore - a.anomalyScore);

//mostramos resultados
/*anomalies.forEach( element => {
    console.log( `${green}${element.anomalyScore}${reset}: ` + testFeatures[element.index]);
});*/

var dataNormal = [];
var dataAnomalies = [];

anomalies.forEach( element => {
    dataAnomalies.push( {x: element.date, y: element.valor } )
});

normal.forEach( element => {
    dataNormal.push( {x: element.date, y: element.valor } )
});

/////////////////////////

// Configuración del lienzo (canvas)
const canvas = createCanvas(800, 600);
const ctx = canvas.getContext('2d');

const chart = new Chart(ctx, {
    type: 'scatter',
    data: {
        datasets: [
            {
                label: 'Datos normales',
                data: dataNormal,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            },
            {
                label: 'Anomalias',
                data: dataAnomalies,
                backgroundColor: 'rgba(153, 102, 255, 0.2)',
                borderColor: 'rgba(153, 102, 255, 1)',
                borderWidth: 1
            },    
        ]
    },
    options: {
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute', // Otras unidades: 'month', 'year', etc.
                    tooltipFormat: 'lll'
                },
                title:{
                    display: true,
                    text: 'fecha'
                }
            },
            y: {
                title: {
                    display: true,
                    text: 'valor'
                }
            }
        }
    }
});

// Guarda el gráfico como una imagen PNG
const buffer = canvas.toBuffer('image/png');
fs.writeFileSync('chart.png', buffer);
console.log('Gráfico guardado en chart.png');