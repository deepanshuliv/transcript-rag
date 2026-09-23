import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Transcript Intelligence — Chat",
  description: "Ask a private interview archive and trace every answer back to its evidence.",
  icons: {
    icon: "/icon.svg",
  },
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
