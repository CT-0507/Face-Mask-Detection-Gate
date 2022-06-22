const mongoose = require('mongoose');
let host = '172.31.250.62';
const EnterLogs = require('../../models/enterLog.js');
var session;
class ChartsController {
    // GET /
    async index(req, res, next) {
        session = req.session;
        if (session.userid) {
            var title = 'Charts';
            var withMask = 0;
            var noMask = 0;
            const data = [];
            var ChartsData;
            try {
                var today = new Date();
                var date = today.getMonth() + '/' + today.getDate() + '/' + today.getFullYear();
                console.log(date)
                try {
                    var Logs = await EnterLogs.find({}, function (err, results) {
                        if (err) throw 'wrong';
                        var d = results.createdAt;
                        if(d.toLocaleDateString() == date){
                            results.forEach(() => {
                                data.push({
                                    withMask: results.withMask,
                                })
                            })
                        }
                    });
                    data.forEach((result) => {
                        if(result.withMask) {
                            withMask++;
                        } else noMask++;
                    })
                    ChartsData = {
                        withMask: withMask,
                        noMask: noMask,
                    }
                    console.log(ChartsData);
                }
                catch(err){
                    console.log(err);
                }
                res.render('charts/show', { ChartsData: ChartsData, title: title, host: host });
            }
            catch (err) {
                res.status(404).send({
                    message: 'error',
                });
            }
        }
        else {
            res.redirect('/');
        }
    }
    async getdata(req, res, next) {
        try {
            const dht = await DHT.findOne().sort({ createdAt: -1 });
            const bh = await BH.findOne().sort({ createdAt: -1 });
            const sl = await SL.findOne().sort({ createdAt: -1 });
            let data = {
                temp: dht.temp,
                humi: dht.humi,
                lux: bh.lux,
                soilHumi: sl.soilHumi,
                tempCreatedAt: dht.createdAt,
                humiCreatedAt: dht.createdAt,
                soilCreatedAt: sl.createdAt,
                luxCreatedAt: bh.createdAt,
            };
            console.log(data);
            res.json({ data: data });
        } catch (err) {
            res.status(404).send({
                message: 'err',
            });
        }
    }
}

module.exports = new ChartsController();
