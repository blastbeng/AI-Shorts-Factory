import "./globals.css";

export const metadata = {
  title: "AI Shorts Factory",
  description: "Autonomous Local AI Video Generation Platform",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body className="bg-gray-900 text-white">{children}</body>
    </html>
  );
}
