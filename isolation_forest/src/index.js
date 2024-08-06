const IsolationForest = require('isolation-forest')
const loadCSVPercentage = require('./loadCSVPercentage')

const { createCanvas } = require('canvas');
const {Chart, registerables} = require('chart.js');
const fs = require('fs');

require('chartjs-adapter-moment');
const moment = require('moment');
// Registrar todos los componentes necesarios
Chart.register(...registerables);

const pathCSV = '../csv/';

// Colores ANSI para console.log
const reset = "\x1b[0m";
const green = "\x1b[32m";

//Cargamos CSV 
const { features , labels, testFeatures, testLabels } = loadCSVPercentage(
    `${pathCSV}prueba.csv`,
    ',',
    {
        shuffle: false,
        percentageTest: 1,
        dataColumns: ['dia de la semana', '0-24','value'],
        labelColumns: ['año','mes','dia','hora','minuto']
    }
)

//creamos isolation forest
var isolationForest = new IsolationForest.IsolationForest();
isolationForest.fit(features) // Type ObjectArray ({}[]); 

var trainingScores = isolationForest.scores()

//predecimos valores con las features
var scores = isolationForest.predict(features)

//creamos array donde colocaremos objetos con el indice y valor de anomalia de los elementos raros.
var anomalies = [];
var normal = [];

console.log("scores", scores.length);
//filtramos los anomalyScore mayores a 0.5
scores.forEach( (element, index) =>{
    if(element > 0.5){  
        
        let value = features[index][2];
        date = new Date( labels[index][0],labels[index][1],labels[index][2],labels[index][3],labels[index][4]*15);
        anomalies.push({"valor": value, "date": date});
    } 
    else {
        let value = features[index][2];
        date = new Date( labels[index][0],labels[index][1],labels[index][2],labels[index][3],labels[index][4]*15);
        normal.push({"valor": value, "date": date});
    }
})

//ordenamos por fechas
anomalies.sort((a, b) => a.date - b.date);
normal.sort((a, b) => a.date - b.date);

//cambiamos las fechas a string
anomalies.map( element => { element.date = moment(element.date).format('YYYY-MM-DD HH:mm')  } );
normal.map( element => { element.date = moment(element.date).format('YYYY-MM-DD HH:mm')  } );

//transformamos los arrays en x,y para poder representarlos en el canvas
var dataNormal = [];
var dataAnomalies = [];

anomalies.forEach( element => {
    dataAnomalies.push( {x: element.date, y: element.valor } )
});

normal.forEach( element => {
    dataNormal.push( {x: element.date, y: element.valor } )
});


// Configuración del lienzo (canvas)
const canvas = createCanvas(5000,5000);
const ctx = canvas.getContext('2d');

const chart = new Chart(ctx, {
    type: 'line',
    data: {
        datasets: [
            {
                label: 'Datos normales',
                data: dataNormal,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1,
                pointRadius: 5,
                pointBackgroundColor: 'rgba(75,192,192,1)',
                showLine: true
            },
            {
                label: 'Anomalias',
                data: dataAnomalies,
                backgroundColor: 'rgba(255, 99, 132, 0.2)',
                borderColor: 'rgba(255, 99, 132, 1)',
                borderWidth: 1,
                pointRadius: 5,
                pointBackgroundColor: 'rgba(255, 99, 132, 1)',
                showLine: true
            },    
        ]
    },
    options: {
        responsive: true,
        maintainAspectRatio: false,
        scales: {
            x: {
                type: 'time',
                time: {
                    unit: 'minute', // Otras unidades: 'month', 'year', etc.
                    displayFormats: {
                        minute: 'MMM DD HH:mm'
                    },
                    tooltipFormat: 'YYYY-MM-DD HH:mm',
                },
                title:{
                    display: true,
                    text: 'fecha'
                },
                ticks:{
                    stepSize:15
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