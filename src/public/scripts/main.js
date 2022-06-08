var socket = io(); // kết nối đến server

var randomScalingFactor = function () {
    return Math.round(Math.random() * 100);
};

var randomData = function () {
    return [
        randomScalingFactor(),
        randomScalingFactor(),
        randomScalingFactor(),
        randomScalingFactor(),
    ];
};

var randomValue = function (data) {
    return Math.max.apply(null, data) * Math.random();
};

var data = randomData();
var value = randomValue(data);

var tempConfig = {
    type: 'gauge',
    data: {
        //labels: ['Success', 'Warning', 'Warning', 'Error'],
        datasets: [
            {
                data: [20, 30, 50],
                value: 20,
                backgroundColor: ['#baabe4', 'green', 'red'],
                borderWidth: 2,
            },
        ],
    },
    options: {
        responsive: true,
        title: {
            display: true,
            text: 'Temperature',
        },
        layout: {
            padding: {
                bottom: 30,
            },
        },
        needle: {
            // Needle circle radius as the percentage of the chart area width
            radiusPercentage: 2,
            // Needle width as the percentage of the chart area width
            widthPercentage: 3.2,
            // Needle length as the percentage of the interval between inner radius (0%) and outer radius (100%) of the arc
            lengthPercentage: 80,
            // The color of the needle
            color: 'rgba(0, 0, 0, 1)',
        },
        valueLabel: {
            formatter: Math.round,
        },
        plugins: {
            datalabels: {
                display: true,
                formatter: function (value, context) {
                    return '< ' + Math.round(value);
                },
                color: function (context) {
                    return context.dataset.backgroundColor;
                },
                //color: 'rgba(255, 255, 255, 1.0)',
                backgroundColor: 'rgba(0, 0, 0, 1.0)',
                borderWidth: 0,
                borderRadius: 5,
                font: {
                    weight: 'bold',
                },
            },
        },
    },
};

var humiConfig = {
    type: 'gauge',
    data: {
        //labels: ['Success', 'Warning', 'Warning', 'Error'],
        datasets: [
            {
                data: [35, 65, 100],
                value: 35,
                backgroundColor: ['red', 'orange', 'green'],
                borderWidth: 2,
            },
        ],
    },
    options: {
        responsive: true,
        title: {
            display: true,
            text: 'Humidity',
        },
        layout: {
            padding: {
                bottom: 30,
            },
        },
        needle: {
            // Needle circle radius as the percentage of the chart area width
            radiusPercentage: 2,
            // Needle width as the percentage of the chart area width
            widthPercentage: 3.2,
            // Needle length as the percentage of the interval between inner radius (0%) and outer radius (100%) of the arc
            lengthPercentage: 80,
            // The color of the needle
            color: 'rgba(0, 0, 0, 1)',
        },
        valueLabel: {
            formatter: Math.round,
        },
        plugins: {
            datalabels: {
                display: true,
                formatter: function (value, context) {
                    return '< ' + Math.round(value);
                },
                color: function (context) {
                    return context.dataset.backgroundColor;
                },
                //color: 'rgba(255, 255, 255, 1.0)',
                backgroundColor: 'rgba(0, 0, 0, 1.0)',
                borderWidth: 0,
                borderRadius: 5,
                font: {
                    weight: 'bold',
                },
            },
        },
    },
};

var lightConfig = {
    type: 'gauge',
    data: {
        //labels: ['Success', 'Warning', 'Warning', 'Error'],
        datasets: [
            {
                data: [200, 300, 1000],
                value: 200,
                backgroundColor: ['red', 'orange', 'green'],
                borderWidth: 2,
            },
        ],
    },
    options: {
        responsive: true,
        title: {
            display: true,
            text: 'Brightness',
        },
        layout: {
            padding: {
                bottom: 30,
            },
        },
        needle: {
            // Needle circle radius as the percentage of the chart area width
            radiusPercentage: 2,
            // Needle width as the percentage of the chart area width
            widthPercentage: 3.2,
            // Needle length as the percentage of the interval between inner radius (0%) and outer radius (100%) of the arc
            lengthPercentage: 80,
            // The color of the needle
            color: 'rgba(0, 0, 0, 1)',
        },
        valueLabel: {
            formatter: Math.round,
        },
        plugins: {
            datalabels: {
                display: true,
                formatter: function (value, context) {
                    return '< ' + Math.round(value);
                },
                color: function (context) {
                    return context.dataset.backgroundColor;
                },
                //color: 'rgba(255, 255, 255, 1.0)',
                backgroundColor: 'rgba(0, 0, 0, 1.0)',
                borderWidth: 0,
                borderRadius: 5,
                font: {
                    weight: 'bold',
                },
            },
        },
    },
};


var soilHumiConfig = {
    type: 'gauge',
    data: {
        //labels: ['Success', 'Warning', 'Warning', 'Error'],
        datasets: [
            {
                data: [35, 80, 100],
                value: 88,
                backgroundColor: ['red', 'orange', 'green'],
                borderWidth: 2,
            },
        ],
    },
    options: {
        responsive: true,
        title: {
            display: true,
            text: 'Soil Humidity',
        },
        layout: {
            padding: {
                bottom: 30,
            },
        },
        needle: {
            // Needle circle radius as the percentage of the chart area width
            radiusPercentage: 2,
            // Needle width as the percentage of the chart area width
            widthPercentage: 3.2,
            // Needle length as the percentage of the interval between inner radius (0%) and outer radius (100%) of the arc
            lengthPercentage: 80,
            // The color of the needle
            color: 'rgba(0, 0, 0, 1)',
        },
        valueLabel: {
            formatter: Math.round,
        },
        plugins: {
            datalabels: {
                display: true,
                formatter: function (value, context) {
                    return '< ' + Math.round(value);
                },
                color: function (context) {
                    return context.dataset.backgroundColor;
                },
                //color: 'rgba(255, 255, 255, 1.0)',
                backgroundColor: 'rgba(0, 0, 0, 1.0)',
                borderWidth: 0,
                borderRadius: 5,
                font: {
                    weight: 'bold',
                },
            },
        },
    },
};

// This code is stupid
function addData(chart, data){
    chart.data.datasets[0].value = data
    chart.update()
}


window.onload = function () {
    var ctx1 = document.getElementById('temp-chart').getContext('2d');
    window.myGauge1 = new Chart(ctx1, tempConfig);
    var ctx2 = document.getElementById('humi-chart').getContext('2d');
    window.myGauge2 = new Chart(ctx2, humiConfig);
    var ctx3 = document.getElementById('light-chart').getContext('2d');
    window.myGauge3 = new Chart(ctx3, lightConfig);
    var ctx4 = document.getElementById('soil-humi-chart').getContext('2d');
    window.myGauge4 = new Chart(ctx4, soilHumiConfig);

    socket.on('sendDhtMain', data =>{
        console.log('dht main')
        addData(window.myGauge1, data.temp)
        addData(window.myGauge2, data.humi)
    })

    socket.on('sendBhMain', data => {
        console.log('bh main')
        addData(window.myGauge3, data.lux)
    })

    socket.on('sendSoilMain', data =>{
        console.log('soil main')
        addData(window.myGauge4, data.soilHumi)
    })

};

//   document.getElementById('randomizeData').addEventListener('click', function() {
//     soilHumiConfig.data.datasets.forEach(function(dataset) {
//       dataset.data = randomData();
//       dataset.value = randomValue(dataset.data);
//     });
//     window.myGauge4.update();
//   });
  //document.addEventListener('DOMContentLoaded', () => {
    // setInterval(() => {
    //     soilHumiConfig.data.datasets.forEach(function(dataset) {
    //         dataset.data = randomData();
    //         dataset.value = randomValue(dataset.data);
    //     });
    //     window.myGauge4.update();
    // }, 3000);
  //})


  const url = `http://localhost:3000/manual`
    const sendingLeds = (url, data) => {
        fetch(url, {
            method: 'POST',
            body: JSON.stringify({data}),
            headers: {"Content-type": "application/json; charset=UTF-8"}
        })
        .then(response => response.json())
        .then(json => console.log(json))
    }

    const clientId = 'webClient';
    const host = 'ws://172.31.251.191:9001';

    var topicPump = 'pump';
    var topicBulb = 'bulb';
    var topicServo = 'servo';

    const options = {
      keepalive: 60,
      clientId: clientId,
      protocolId: 'MQTT',
      protocolVersion: 4,
      clean: true,
      reconnectPeriod: 1000,
      connectTimeout: 30 * 1000,
      will: {
          topic: 'WillMsg',
          payload: 'Connection Closed abnormally..!',
          qos: 0,
          retain: false
      },
    }

    console.log('Connecting mqtt client')

    const client = mqtt.connect(host, options)

    client.on('error', (err) => {
      console.log('Connection error: ', err)
      client.end()
    })


    client.on('reconnect', () => {
      console.log('Reconnecting...')
    })

    client.on('connect', ()=>{
      console.log('Client connected')
      // client.publish('test', 'Web browser connect')
    })

    document.addEventListener('DOMContentLoaded', function() {
        $("input[type=checkbox]#pump").change(function(){
            if ( this.checked ){
                var data = JSON.stringify({pump:true})
                client.publish(topicPump, data);
                   
            } else {
                var data = JSON.stringify({pump:false})
                client.publish(topicPump, data)
            }
        })

        $("input[type=checkbox]#bulb").change(function(){
            if ( this.checked ){
                var data = JSON.stringify({servo: true})
                client.publish(topicServo, data);
            }
            else {
                var data = JSON.stringify({servo: false})
                client.publish(topicServo, data)
            }
        });
    
});
