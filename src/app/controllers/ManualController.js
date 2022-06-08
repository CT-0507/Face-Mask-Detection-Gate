const mongoose = require('mongoose');
const User = require('../../models/user.js');
var session;
class ManualController {
    publish(req, res, next) {
        session = req.session;
        if (session.userid) {
            console.log('OK');
            res.status(200).send({
                message: 'OK',
            });
        } else
            res.status(404).send({
                message: 'error',
            });
    }
}

module.exports = new ManualController();
