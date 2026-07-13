import './globals.css';

export const metadata = {
  title: 'Aabha — Accessible AI Image Caption & Audio Narrator',
  description: 'Aabha - An accessible web app that instantly translates uploaded images into descriptive audio, empowering visually impaired users to perceive visual content effortlessly.',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <head>
        <link rel="preconnect" href="https://fonts.googleapis.com" />
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="anonymous" />
        <link href="https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Inter:wght@400;500;600;700&family=Outfit:wght@500;600;700;800&display=swap" rel="stylesheet" />
      </head>
      <body>
        {children}
      </body>
    </html>
  );
}
