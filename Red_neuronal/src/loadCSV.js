const fs = require('fs')
const _ = require('lodash')
const shuffleSeed = require('shuffle-seed')

// Función para quitar las comillas dobles de los nombres de columna
function stripQuotes(columnName) {
    return columnName.replace(/^"(.*)"$/, '$1');
}

//funcion generica para quitar cadenas a string, para ello usamos expresiones regulares
function extractCharacter(columnName,regular, characterReplace) {
    return columnName.replace(regular, characterReplace);
}

//con este metodo extraemos los valores que nos ha pasado el usuario y los dividimos en input y output
function extractColumns(data, columnNames){
    const headers = _.first(data);

    const indexes = _.map(columnNames, column => headers.indexOf(column) );

    const extracted = _.map(data, row => _.pullAt(row, indexes) );

    return extracted;
}

module.exports = function loadCSV(
    filename, 
    characterSplit,
    {
        dataColumns     = [],   //indicar nombre de las columnas que son los inputs que se usan para predecir

        labelColumns    = [],   //indicar nombre de la columna que es el resultado a predecir

        shuffle         = true, //por si quieres mezclar los valores

        splitTest       = 0,    //desde que valor quieres que sean para test. Si pones 5, en el csv a partir de la fila 6 el resto seran para hacer test
                                // 10       -> valor minimo para que funcione bien
                                // false    -> si no quieres valores para test 

        converters      = {}    //añadir conversiones de columnas a valores
    } 
) {
    
    //leemos el fichero csv
    let data = fs.readFileSync(filename, { encoding: 'utf-8' });
    
    //separamos el fichero por filas y por columnas segun el caracter que le pasamos a la funcion
    data = data.split('\n').map(row => row.split(characterSplit));

    //esto nos sirve por si se genera el csv con campos al final vacias value,id,,,, -> value,id
    data = data.map(row => _.dropRightWhile(row, val => val === '') );

    //console.log("divididos por filas y columnas: ", data);

    //quitamos los \r de algunas cadenas
    data = data.map( (row, indexRow) => {
        return row.map( (column, indexColumn) =>{
            return extractCharacter(column,/\r/g, '');
        })
    })

    //console.log("despues de quitar \\r: ", data);


    //quitamos de los titulos '"value"', las ""
    for (let i = 0; i < data[0].length; i++) {
        data[0][i] = stripQuotes(data[0][i]);
    }

    //console.log("quitamos \"\" : ", data);
    

    const headers = _.first(data);

    data = data.map( (row, index) => {

        //en la primera fila estan los nombres
        if(index === 0) return row;

        return row.map( (element, index) => {

            //en caso de tener que convertir algun dato, lo hacemos segun nos indican los converters
            if(converters[headers[index]]){
                const converted = converters[headers[index]](element);
                return _.isNaN(converted) ? element : converted
            }

            const result = parseFloat(element);
            return _.isNaN(result) ? element : result; 
        });

    });

    //console.log("Despues de pasar a float: ", data);

    //separamos en distintos arrays los labels y features
    let labels = extractColumns(data, labelColumns);
    data = extractColumns(data, dataColumns);

    //console.log("features separadas", data);
    //console.log("label", labels);

    //quitamos el primer elemento de cada array ya que estan los nombres de las columnas y no lo queremos
    labels.shift();
    data.shift();

    //en caso de indicar de mezclar el array, lo mezclamos
    if( shuffle ){
        data = shuffleSeed.shuffle(data, 'phrase' ); //ambos array se tienen que mezclar con el mismo string, por eos ponemos phrase en ambos
        labels = shuffleSeed.shuffle(labels, 'phrase' );
    }

    //en caso de querer hacer test, dividimos las salidas en valores de entrenamiento y test
    if( splitTest ){
        const trainSize = _.isNumber( splitTest ) ? splitTest : Math.floor(data.length / 2);

        return {
            features:       data.slice(0,trainSize),
            labels:         labels.slice(0,trainSize),
            testFeatures:   data.slice(trainSize),
            testLabels:     labels.slice(trainSize),
        }

    }
    //si no se quiere hacer test solo hay valores de entrenamiento
    else{
        return { features: data, labels };
    }


}
    
