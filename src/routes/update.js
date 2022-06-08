const express = require('express');
const router = express.Router();

const updateController = require('../app/controllers/UpdateController');

router.post('/device', updateController.addDevice); // /update/device with json file
router.post('/dht', updateController.updateDht);
router.post('/bh', updateController.updateBh);
router.post('/soil', updateController.updateSoil);
module.exports = router;
