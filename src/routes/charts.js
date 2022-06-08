const express = require('express');
const router = express.Router();

const chartsController = require('../app/controllers/ChartsController');

router.get('/get-data', chartsController.getdata);
router.get('/', chartsController.index);

module.exports = router;
