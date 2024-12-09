const express = require('express');
const nodemailer = require('nodemailer');
const dotenv = require('dotenv');
const fs = require('fs');
const path = require('path');

dotenv.config();

const app = express();
const port = 3000;

// Set up email transporter using nodemailer
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: 'navya0807@gmail.com',  // Your email address
        pass: 'imco rsxd ultc wuqm'   // Your email password or app-specific password
    }
});

// Middleware to parse JSON bodies
app.use(express.json());

// POST endpoint to send email
app.post('/send-email', (req, res) => {
    const { subject, text, to } = req.body;

    // Check for recipient
    if (!to) {
        return res.status(400).send('No recipient defined.');
    }

    // Set up email options
    const mailOptions = {
        from: 'navya0807@gmail.com',  // Sender address
        to: to,                       // Recipient address
        subject: subject,             // Email subject
        text: text,                   // Email body
    };

    // Send email
    transporter.sendMail(mailOptions, (error, info) => {
        if (error) {
            console.error('Error sending email:', error);
            return res.status(500).send('Failed to send email.');
        }
        console.log('Email sent:', info.response);
        return res.status(200).send('Email sent successfully.');
    });
});

// Start the server
app.listen(port, () => {
    console.log(`Email service listening at http://localhost:${port}`);
});
