const IsolationForest = require('ml-isolation-forest');

const { createCanvas } = require('canvas');
const {Chart, registerables} = require('chart.js');
const fs = require('fs');

// Registrar todos los componentes necesarios
Chart.register(...registerables);

//const xl = require('excel4node');

let X = [
  [200, 50],
  [0.3, 0.1],
  [0.5, 0.3],
  [0.2, 0.1],
  [0.1, 0.1],
  [0.2, 0.05],
  [0.3, 0.3],
  [0.4, 0.2],
  [0.3, 0.4],
  [0.1, 0.1],
  [0.05, 0.1],
];

let anomalyDetector = new IsolationForest.IsolationForest();

anomalyDetector.train(X);

let result = anomalyDetector.predict([
  [200, 300],
  [0, 0.1],
  [0.2, 0.1],
  [0.1, 0.2],
]);

console.log(result);

var anomalias = [];
var normales  = []

//filtramos los anomalyScore mayores a 0.5
result.forEach( (element, index) =>{
  if(element > 0.5) anomalias.push( X[index] );
  else normales.push( X[index] );
})


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
                data: normales,
                backgroundColor: 'rgba(75, 192, 192, 0.2)',
                borderColor: 'rgba(75, 192, 192, 1)',
                borderWidth: 1
            },
            {
              label: 'Anomalias',
              data: anomalias,
              backgroundColor: 'rgba(153, 102, 255, 0.2)',
              borderColor: 'rgba(153, 102, 255, 1)',
              borderWidth: 1
          },   
        ]
    }
});

// Guarda el gráfico como una imagen PNG
const buffer = canvas.toBuffer('image/png');
fs.writeFileSync('chart.png', buffer);
console.log('Gráfico guardado en chart.png');

//////////////////////////



/////////////////////////
/*
//Creamos el Excel
var wb = new xl.Workbook();

//Ponemos la fecha
let date = new Date();
let fechaDia    = date.getUTCDate();
let fechaMes    = (date.getUTCMonth()) + 1; 
let fechaAño    = date.getUTCFullYear();

//Ponemos el nombre del archivo
let nombreArchivo = "todosUsuarios" + fechaDia + "_" + fechaMes + "_" + fechaAño + ".";
var ws = wb.addWorksheet(nombreArchivo);

//Creamos la ruta del excel
const path = require('path');
const pathExcel = path.join(__dirname, 'excel', nombreArchivo + '.xlsx');

//Escribir o guardar
wb.write(pathExcel, function(err, stats){
  if(err) console.log(err);
});
*/
/////////////////////////