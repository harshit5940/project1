export const metadata = {
  title: 'Churn Dashboard',
  description: 'Customer Churn Dashboard (Next.js + FastAPI)',
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body style={{ fontFamily: 'sans-serif', margin: 0, padding: 20, background: '#0b1220', color: '#e6e6e6' }}>
        <main style={{ maxWidth: 1200, margin: '0 auto' }}>{children}</main>
      </body>
    </html>
  );
}
