import { defineConfig } from "astro/config";
import starlight from "@astrojs/starlight";
import tailwindcss from "@tailwindcss/vite";

// Project site lives at https://macsermkiat.github.io/blood_audit/
// `site` + `base` together control absolute asset URLs in the built output.
export default defineConfig({
  site: "https://macsermkiat.github.io",
  base: "/blood_audit",
  trailingSlash: "ignore",
  vite: {
    plugins: [tailwindcss()],
  },
  integrations: [
    starlight({
      title: "blood_audit",
      description:
        "KCMH RBC transfusion audit pipeline — post-hoc quality-improvement audit against PR 17.2 + AABB 2023.",
      defaultLocale: "en",
      locales: {
        en: { label: "English", lang: "en" },
        th: { label: "ไทย", lang: "th" },
      },
      social: [
        {
          icon: "github",
          label: "GitHub",
          href: "https://github.com/macsermkiat/blood_audit",
        },
      ],
      customCss: ["./src/styles/custom.css"],
      sidebar: [
        {
          label: "Operators",
          translations: { th: "ผู้ดูแลระบบ" },
          items: [
            { label: "Overview", translations: { th: "ภาพรวม" }, slug: "operators/overview" },
            { label: "Install", translations: { th: "ติดตั้ง" }, slug: "operators/install" },
            { label: "Environment variables", translations: { th: "ตัวแปรสภาพแวดล้อม" }, slug: "operators/env-vars" },
            { label: "First audit run", translations: { th: "รันออดิทครั้งแรก" }, slug: "operators/first-audit-run" },
            { label: "Run the pipeline", translations: { th: "รัน pipeline" }, slug: "operators/run-pipeline" },
            { label: "Ingest CSV", translations: { th: "Ingest CSV" }, slug: "operators/ingest-csv" },
            { label: "audit command", translations: { th: "คำสั่ง audit" }, slug: "operators/audit-command" },
            { label: "Integration seams", translations: { th: "จุดเชื่อมต่อระบบ" }, slug: "operators/integration-seams" },
            { label: "Troubleshooting", translations: { th: "แก้ปัญหาเบื้องต้น" }, slug: "operators/troubleshooting" },
          ],
        },
        {
          label: "Developers",
          translations: { th: "นักพัฒนา" },
          items: [
            { label: "Architecture", translations: { th: "สถาปัตยกรรม" }, slug: "developers/architecture" },
            { label: "Pipeline stages", translations: { th: "ขั้นตอน pipeline" }, slug: "developers/pipeline-stages" },
            { label: "3-tier Hb classifier", translations: { th: "ตัวจำแนก 3-tier Hb" }, slug: "developers/three-tier-hb" },
            { label: "Quote grounding (6 layers)", translations: { th: "Quote grounding 6 ชั้น" }, slug: "developers/quote-grounding" },
            { label: "De-identification", translations: { th: "De-identification" }, slug: "developers/deid" },
            { label: "Audit store", translations: { th: "Audit store" }, slug: "developers/audit-store" },
            { label: "Contributing", translations: { th: "การ contribute" }, slug: "developers/contributing" },
            { label: "Testing", translations: { th: "การทดสอบ" }, slug: "developers/testing" },
          ],
        },
        {
          label: "Clinical overview",
          translations: { th: "ภาพรวมเชิงคลินิก" },
          items: [
            { label: "What this is", translations: { th: "นี่คืออะไร" }, slug: "clinical/what-this-is" },
            { label: "Policy summary", translations: { th: "สรุป policy" }, slug: "clinical/policy-summary" },
            { label: "Report output", translations: { th: "ผลลัพธ์รายงาน" }, slug: "clinical/report-output" },
          ],
        },
        {
          label: "Reference",
          translations: { th: "อ้างอิง" },
          items: [
            { label: "Module glossary", translations: { th: "Glossary ของ modules" }, slug: "reference/module-glossary" },
            { label: "Report schema", translations: { th: "Report schema" }, slug: "reference/report-schema" },
            { label: "Changelog", translations: { th: "Changelog" }, slug: "reference/changelog" },
          ],
        },
      ],
    }),
  ],
});
