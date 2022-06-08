const mongoose = require('mongoose');
const User = require('../../models/user.js');
const Device = require('../../models/devices.js');
var session;
class DashboardController {
    async index(req, res, next) {
        session = req.session;
        if (session.userid) {
            const title = 'Dashboard';
        // Lấy dữ liệu từ database
        const devicesInfo = await Device.find();
        const devices = [];
        devicesInfo.forEach((result) => {
            let isAlive = true;
            let timeLeft =
                (new Date().getTime() - result.updatedAt.getTime()) / 1000;
            if (timeLeft > 300) {
                isAlive = false;
            }
            let updatedAt = result.updatedAt.toLocaleDateString();
            devices.push({
                name: result.name,
                imgUrl: result.imgUrl,
                updatedAt: updatedAt,
                isAlive: isAlive,
            });
        });
        // var tittle = 'Dashboard';
        res.render('dashboard/show', { 
            devices: devices, 
            title: title 
        });
        } else res.redirect('/');
    }
}

module.exports = new DashboardController();
