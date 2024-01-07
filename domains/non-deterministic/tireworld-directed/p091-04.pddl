(define (problem tireworld-091-04)
    (:domain tire-adl)
    (:requirements :negative-preconditions :non-deterministic :strips :typing)
    (:objects l0 l1 l10 l11 l12 l13 l14 l15 l16 l17 l18 l19 l2 l20 l21 l22 l23 l24 l25 l26 l27 l28 l29 l3 l30 l31 l32 l33 l34 l35 l36 l37 l38 l39 l4 l40 l41 l42 l43 l44 l45 l46 l47 l48 l49 l5 l50 l51 l52 l53 l54 l55 l56 l57 l58 l59 l6 l60 l61 l62 l63 l64 l65 l66 l67 l68 l69 l7 l70 l71 l72 l73 l74 l75 l76 l77 l78 l79 l8 l80 l81 l82 l83 l84 l85 l86 l87 l88 l89 l9 l90 - location)
    (:init (road l0 l32) (road l0 l47) (road l1 l6) (road l1 l63) (road l10 l68) (road l10 l78) (road l10 l82) (road l11 l12) (road l11 l44) (road l12 l11) (road l12 l58) (road l13 l64) (road l13 l83) (road l14 l23) (road l14 l8) (road l15 l17) (road l15 l84) (road l16 l20) (road l16 l52) (road l17 l15) (road l17 l34) (road l18 l53) (road l18 l72) (road l19 l24) (road l19 l36) (road l2 l41) (road l2 l89) (road l20 l51) (road l20 l61) (road l21 l48) (road l21 l57) (road l22 l47) (road l23 l14) (road l24 l19) (road l24 l63) (road l25 l28) (road l25 l4) (road l26 l30) (road l26 l77) (road l27 l54) (road l27 l87) (road l28 l25) (road l28 l61) (road l29 l30) (road l29 l88) (road l3 l59) (road l3 l84) (road l30 l29) (road l31 l46) (road l32 l0) (road l32 l33) (road l33 l32) (road l33 l53) (road l34 l17) (road l34 l41) (road l35 l23) (road l36 l60) (road l37 l22) (road l37 l64) (road l38 l57) (road l38 l90) (road l39 l60) (road l39 l86) (road l4 l25) (road l4 l67) (road l40 l73) (road l40 l90) (road l41 l2) (road l41 l34) (road l42 l56) (road l42 l85) (road l43 l49) (road l43 l79) (road l44 l11) (road l44 l70) (road l45 l53) (road l45 l81) (road l46 l31) (road l46 l80) (road l47 l0) (road l47 l22) (road l47 l7) (road l47 l71) (road l48 l21) (road l48 l59) (road l49 l43) (road l49 l68) (road l49 l75) (road l5 l35) (road l5 l85) (road l50 l87) (road l51 l20) (road l51 l72) (road l52 l16) (road l52 l66) (road l53 l18) (road l53 l33) (road l53 l45) (road l54 l27) (road l54 l65) (road l55 l85) (road l55 l89) (road l56 l42) (road l56 l78) (road l57 l21) (road l57 l38) (road l58 l12) (road l58 l67) (road l59 l3) (road l59 l48) (road l6 l1) (road l6 l80) (road l60 l36) (road l60 l39) (road l61 l20) (road l61 l28) (road l62 l50) (road l62 l74) (road l63 l1) (road l63 l24) (road l64 l13) (road l64 l37) (road l65 l54) (road l65 l76) (road l66 l52) (road l66 l8) (road l67 l4) (road l67 l58) (road l68 l10) (road l68 l49) (road l68 l9) (road l69 l71) (road l69 l77) (road l69 l82) (road l7 l47) (road l7 l9) (road l70 l31) (road l70 l44) (road l71 l47) (road l71 l69) (road l72 l18) (road l72 l51) (road l73 l40) (road l73 l76) (road l74 l62) (road l74 l75) (road l75 l49) (road l75 l74) (road l76 l65) (road l76 l73) (road l77 l26) (road l77 l69) (road l78 l10) (road l78 l56) (road l79 l43) (road l8 l14) (road l8 l66) (road l80 l6) (road l81 l45) (road l81 l88) (road l82 l10) (road l82 l69) (road l83 l13) (road l83 l86) (road l84 l15) (road l84 l3) (road l85 l42) (road l85 l5) (road l85 l55) (road l86 l39) (road l86 l83) (road l87 l27) (road l88 l29) (road l88 l81) (road l89 l2) (road l89 l55) (road l9 l7) (road l90 l38) (road l90 l40) (spare-in l0) (spare-in l1) (spare-in l10) (spare-in l12) (spare-in l13) (spare-in l14) (spare-in l15) (spare-in l16) (spare-in l17) (spare-in l18) (spare-in l2) (spare-in l20) (spare-in l21) (spare-in l22) (spare-in l23) (spare-in l24) (spare-in l25) (spare-in l26) (spare-in l27) (spare-in l28) (spare-in l29) (spare-in l3) (spare-in l30) (spare-in l32) (spare-in l33) (spare-in l34) (spare-in l35) (spare-in l36) (spare-in l38) (spare-in l39) (spare-in l4) (spare-in l40) (spare-in l41) (spare-in l42) (spare-in l43) (spare-in l44) (spare-in l45) (spare-in l48) (spare-in l49) (spare-in l5) (spare-in l50) (spare-in l51) (spare-in l52) (spare-in l53) (spare-in l54) (spare-in l55) (spare-in l57) (spare-in l58) (spare-in l59) (spare-in l6) (spare-in l61) (spare-in l62) (spare-in l63) (spare-in l65) (spare-in l66) (spare-in l67) (spare-in l68) (spare-in l69) (spare-in l72) (spare-in l73) (spare-in l74) (spare-in l75) (spare-in l76) (spare-in l77) (spare-in l78) (spare-in l8) (spare-in l80) (spare-in l81) (spare-in l82) (spare-in l83) (spare-in l84) (spare-in l85) (spare-in l86) (spare-in l87) (spare-in l89) (spare-in l9) (spare-in l90) (vehicle-at l79))
    (:goal (vehicle-at l11))
)