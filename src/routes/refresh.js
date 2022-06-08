const express = require('express');
const router = express.Router();

const RefreshController = require('../app/controllers/RefreshController.js');

router.patch('/', RefreshController.refreshDevice);
module.exports = router;
