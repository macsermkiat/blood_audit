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
            { label: "Ingest CSV", slug: "operators/ingest-csv" },
            { label: "audit command", slug: "operators/audit-command" },
            { label: "Integration seams", slug: "operators/integration-seams" },
            { label: "Troubleshooting", slug: "operators/troubleshooting" },
          ],
        },
        {
          label: "Developers",
          items: [
            { label: "Architecture", slug: "developers/architecture" },
            { label: "Pipeline stages", slug: "developers/pipeline-stages" },
            { label: "3-tier Hb classifier", slug: "developers/three-tier-hb" },
            { label: "Quote grounding (6 layers)", slug: "developers/quote-grounding" },
            { label: "De-identification", slug: "developers/deid" },
            { label: "Audit store", slug: "developers/audit-store" },
            { label: "Contributing", slug: "developers/contributing" },
            { label: "Testing", slug: "developers/testing" },
          ],
        },
        {
          label: "Clinical overview",
          translations: { th: "ภาพรวมเชิงคลินิก" },
          items: [
            { label: "What this is", translations: { th: "นี่คืออะไร" }, slug: "clinical/what-this-is" },
            { label: "Policy summary", slug: "clinical/policy-summary" },
            { label: "Report output", slug: "clinical/report-output" },
          ],
        },
        {
          label: "Reference",
          items: [
            { label: "Module glossary", slug: "reference/module-glossary" },
            { label: "Report schema", slug: "reference/report-schema" },
            { label: "Changelog", slug: "reference/changelog" },
          ],
        },
      ],
    }),
  ],
});
