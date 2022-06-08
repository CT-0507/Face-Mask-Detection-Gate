const mongoose = require('mongoose');
const User = require('../../models/user.js');
var session;
class DashboardController {
    logout(req, res, next) {
        req.session.destroy();
        res.redirect('/');
    }
}

module.exports = new DashboardController();
