const mongoose = require('mongoose');
const DHT = require('../../models/dht.js');
const BH = require('../../models/bh.js');
const SL = require('../../models/soil.js');
//var { Chart } = require('charts.js');
//import 'chartjs-adapter-luxon';
//var ChartStreaming = require('chartjs-plugin-streaming');

//Chart.register(ChartStreaming);
let host = "172.31.250.62"
class ChartsController {
    // GET /
    async index(req, res, next) {
        var title = 'Charts';
        const dht = await DHT.findOne().sort({ updatedAt: -1 });
        const bh = await BH.findOne().sort({ updatedAt: -1 });
        const sl = await SL.findOne().sort({updatedAt: -1});
        let data = {
            temp: dht.temp,
            humi: dht.humi,
            lux: bh.lux,
            soildHumi: sl.soildHumi,
        };
        res.render('charts/show', { data: data, title: title, host: host });
    }
    async getdata(req, res, next) {
        const dht = await DHT.findOne().sort({ createdAt: -1 });
        const bh = await BH.findOne().sort({ createdAt: -1 });
        const sl = await sl.findOne().sort({ createdAt: -1 });
        let data = {
            temp: dht.temp,
            humi: dht.humi,
            lux: bh.lux,
            soildHumi: sl.soildHumi,
            tempCreateAt: dht.createdAt,
            humiCreateAt: dht.createdAt,
            soilCreateAt: sl.createdAt,
            luxCreateAt: bh.createdAt,
        };
        console.log(data);
        res.json({ data: data });
    }
}

module.exports = new ChartsController();