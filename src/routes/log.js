const express = require('express');
const router = express.Router();

const logController = require('../app/controllers/LogController');

router.delete('/delete-all-log', logController.destroyAll);
router.delete('/:id', logController.destroy);
router.get('/filter', logController.filter);
router.get('/filter/:page', logController.filter);
router.get('/', logController.index);

module.exports = router;
