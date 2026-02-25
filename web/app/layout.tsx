import type { Metadata } from "next";
import type { ReactNode } from "react";
import "@/app/globals.css";

export const metadata: Metadata = {
  title: "CarCompose MVP",
  description: "One-click car compositing pipeline"
};

export default function RootLayout({
  children
}: Readonly<{
  children: ReactNode;
}>): JSX.Element {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
