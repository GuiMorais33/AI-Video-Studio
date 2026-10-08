import type { Metadata } from "next";
import Link from "next/link";
import "./globals.css";

export const metadata: Metadata = {
  title: "AI Video Studio",
  description: "Transformação de vídeos reais com IA — rastreamento com SAM 2.1",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="pt-BR">
      <body>
        <header className="topbar">
          <Link href="/" className="brand">
            AI Video Studio
          </Link>
          <span className="phase">Fase 2 · Seleção e rastreamento (SAM 2.1)</span>
        </header>
        <main className="container">{children}</main>
      </body>
    </html>
  );
}
