"use client";
import { useEffect, useState } from "react";
import {
  ResponsiveContainer,
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || "http://localhost:8000";

export default function Home() {
  const [kpis, setKpis] = useState(null);
  const [contractAgg, setContractAgg] = useState([]);
  const [paymentAgg, setPaymentAgg] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    async function load() {
      try {
        const [k, c, p] = await Promise.all([
          fetch(`${API_BASE}/kpis`).then((r) => r.json()),
          fetch(`${API_BASE}/aggregates/contract`).then((r) => r.json()),
          fetch(`${API_BASE}/aggregates/payment_method`).then((r) => r.json()),
        ]);
        setKpis(k);
        setContractAgg(c);
        setPaymentAgg(p);
      } catch (e) {
        setError("Failed to load from API. Set NEXT_PUBLIC_API_BASE to your backend URL.");
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  return (
    <div>
      <h1 style={{ marginBottom: 8 }}>Customer Churn Dashboard</h1>
      <p style={{ opacity: 0.8, marginTop: 0 }}>Next.js frontend on Vercel + FastAPI backend</p>

      {loading && <p>Loading...</p>}
      {error && <p style={{ color: "#ff6b6b" }}>{error}</p>}

      {kpis && (
        <div style={{ display: "grid", gridTemplateColumns: "repeat(5, 1fr)", gap: 12, marginTop: 20 }}>
          <KpiCard label="Churn Rate" value={`${kpis.churn_rate.toFixed(1)}%`} />
          <KpiCard label="Retention Rate" value={`${kpis.retention_rate.toFixed(1)}%`} />
          <KpiCard label="Avg Tenure" value={`${(kpis.avg_tenure || 0).toFixed(1)} mo`} />
          <KpiCard label="High-Risk Share" value={`${kpis.high_risk_share.toFixed(1)}%`} />
          <KpiCard label="CLV (avg)" value={`$${(kpis.clv_estimate || 0).toFixed(0)}`} />
        </div>
      )}

      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr", gap: 20, marginTop: 24 }}>
        <ChartCard title="Churn Rate by Contract Type">
          <BarResponsive data={contractAgg} xKey="Contract" yKey="churn_rate" />
        </ChartCard>
        <ChartCard title="Churn Rate by Payment Method">
          <BarResponsive data={paymentAgg} xKey="PaymentMethod" yKey="churn_rate" />
        </ChartCard>
      </div>

      <div style={{ marginTop: 24 }}>
        <p style={{ fontSize: 12, opacity: 0.7 }}>
          API Base: <code>{API_BASE}</code>
        </p>
      </div>
    </div>
  );
}

function KpiCard({ label, value }) {
  return (
    <div style={{ background: "#121a2b", padding: 16, borderRadius: 12, border: "1px solid #1f2a44" }}>
      <div style={{ opacity: 0.7, fontSize: 13 }}>{label}</div>
      <div style={{ fontSize: 22, marginTop: 6 }}>{value}</div>
    </div>
  );
}

function ChartCard({ title, children }) {
  return (
    <div style={{ background: "#121a2b", padding: 16, borderRadius: 12, border: "1px solid #1f2a44", height: 360 }}>
      <div style={{ marginBottom: 8, opacity: 0.85 }}>{title}</div>
      <div style={{ width: "100%", height: 300 }}>{children}</div>
    </div>
  );
}

function BarResponsive({ data, xKey, yKey }) {
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={data} margin={{ top: 10, right: 10, left: 0, bottom: 40 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="#233053" />
        <XAxis dataKey={xKey} stroke="#93a4c3" angle={-25} textAnchor="end" interval={0} height={60} />
        <YAxis stroke="#93a4c3" />
        <Tooltip contentStyle={{ backgroundColor: "#1b2540", border: "1px solid #2a3a61" }} />
        <Bar dataKey={yKey} fill="#6aa0ff" />
      </BarChart>
    </ResponsiveContainer>
  );
}
