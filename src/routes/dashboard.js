const express = require('express');
const router = express.Router();

const dashboardController = require('../app/controllers/DashboardController');

router.delete('delete/:id', dashboardController.destroy);
router.get('/', dashboardController.index);

module.exports = router;
