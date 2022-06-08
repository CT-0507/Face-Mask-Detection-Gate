const express = require('express');
const router = express.Router();

const dashboardController = require('../app/controllers/DashboardController');

router.get('/', dashboardController.index);

module.exports = router;
