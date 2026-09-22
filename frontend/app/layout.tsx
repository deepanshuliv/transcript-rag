import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "Transcript Intelligence",
  description: "Grounded question answering over European robotic surgery interviews.",
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
