const mongoose = require('mongoose');

const UserSchema = new mongoose.Schema({
    name: String,
    email: { type: String, unique: true, required: true },
    password: String,
    role: { type: String, enum: ['employee', 'hr'], required: true }
});

module.exports = mongoose.model('User', UserSchema);
