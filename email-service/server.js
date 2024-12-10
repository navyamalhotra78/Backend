const express = require('express');
const nodemailer = require('nodemailer');
const multer = require('multer'); // For handling file uploads

const app = express();
const port = 3000;

// Set up email transporter using nodemailer
const transporter = nodemailer.createTransport({
    service: 'gmail',
    auth: {
        user: 'navya0807@gmail.com', // Hardcoded email address
        pass: 'imco rsxd ultc wuqm'  // Hardcoded app password
    }
});

// Middleware to handle JSON and URL-encoded form data
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Set up multer for file uploads
const upload = multer();

// POST endpoint to send email
app.post('/send-email', upload.single('attachment'), (req, res) => {
    const { subject, text, to } = req.body;
    const attachment = req.file;

    console.log('Received email data:', req.body);
    if (attachment) {
        console.log('Attachment details:', attachment.originalname);
    } else {
        console.log('No attachment provided.');
    }

    if (!to) {
        return res.status(400).send('No recipient defined.');
    }

    // Set up email options
    const mailOptions = {
        from: 'navya0807@gmail.com',
        to,
        subject,
        text,
        attachments: attachment ? [{
            filename: attachment.originalname,
            content: attachment.buffer // File content
        }] : [] // Empty if no attachment
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
